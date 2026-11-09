// npz_archive.hpp —— 类实现的基础设施：线程池、NPY/NPZ 归档、并行稠密矩阵
// ============================================================================
// 仅依赖 C++11 标准库 + zlib（Strawberry 自带）。
//   · ThreadPool ：std::thread 工作线程 + parallel_for
//   · NpyArray   ：NPY 头解析/生成（支持 <f8 / <i8 / <i4 / |u1）
//   · NpzArchive ：真正的 .npz（ZIP 容器，raw-deflate via zlib）读写
//   · DMat       ：行优先稠密矩阵，矩阵乘用线程池并行
// 论文：《多尺度指纹级联框架 QuadFP-Cascade》（Computer Communications 251 (2026) 108495）
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <zlib.h>

namespace qpfc {

// ----------------------------------------------------------------------
// 线程池：固定数量工作线程，支持 parallel_for 并等待全部完成
// ----------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(int nthreads) : stop_(false) {
        if (nthreads < 1) nthreads = 1;
        for (int i = 0; i < nthreads; ++i)
            workers_.emplace_back([this] { this->worker_loop(); });
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lk(mtx_); stop_ = true; }
        cv_.notify_all();
        for (size_t i = 0; i < workers_.size(); ++i)
            if (workers_[i].joinable()) workers_[i].join();
    }
    int size() const { return (int)workers_.size(); }

    void submit(std::function<void()> job) {
        { std::unique_lock<std::mutex> lk(mtx_); tasks_.push(job); }
        cv_.notify_one();
    }

    // 对 [begin,end) 每个整数执行 body(i)，分块并行并阻塞等待全部完成
    void parallel_for(int begin, int end,
                      const std::function<void(int)>& body) {
        if (end <= begin) return;
        long long total = (long long)(end - begin);
        int chunks = (int)std::min<long long>(total, (long long)workers_.size());
        Latch latch(chunks);
        long long per = total / chunks, rem = total % chunks, cur = begin;
        for (int q = 0; q < chunks; ++q) {
            long long cnt = per + (q < rem ? 1 : 0);
            long long b0 = cur, b1 = cur + cnt; cur = b1;
            submit([body, b0, b1, &latch]() {
                for (long long i = b0; i < b1; ++i) body((int)i);
                latch.done();
            });
        }
        latch.wait();
    }

private:
    struct Latch {
        int count; std::mutex m; std::condition_variable cv;
        explicit Latch(int n) : count(n) {}
        void done() {
            std::unique_lock<std::mutex> lk(m);
            --count;
            if (count == 0) cv.notify_all();
        }
        void wait() {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [this] { return count == 0; });
        }
    };
    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            { std::unique_lock<std::mutex> lk(mtx_);
              cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
              if (stop_ && tasks_.empty()) return;
              job = tasks_.front(); tasks_.pop();
            }
            job();
        }
    }
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_;
};

// ----------------------------------------------------------------------
// NPY 数组：解析与生成
// ----------------------------------------------------------------------
class NpyArray {
public:
    std::string dtype;
    std::vector<long long> shape;
    bool isfloat;
    std::vector<double> f;
    std::vector<long long> i;

    NpyArray() : isfloat(false) {}

    long long numel() const {
        long long n = 1;
        for (size_t k = 0; k < shape.size(); ++k) n *= shape[k];
        return n;
    }

    static NpyArray from_f64(std::vector<long long> sh,
                             const std::vector<double>& data) {
        NpyArray a; a.dtype = "<f8"; a.shape = sh;
        a.isfloat = true; a.f = data; return a;
    }
    static NpyArray from_i64(std::vector<long long> sh,
                             const std::vector<long long>& data) {
        NpyArray a; a.dtype = "<i8"; a.shape = sh;
        a.isfloat = false; a.i = data; return a;
    }

    static NpyArray parse(const std::vector<unsigned char>& raw) {
        NpyArray a;
        if (raw.size() < 10 ||
            !(raw[0] == 0x93 && raw[1] == 'N' && raw[2] == 'U' &&
              raw[3] == 'M' && raw[4] == 'P' && raw[5] == 'Y'))
            throw std::runtime_error("bad NPY magic");
        unsigned hlen = (unsigned)(raw[8] | (raw[9] << 8));
        std::string header((const char*)raw.data() + 10, hlen);
        a.dtype = quoted_after(header, "descr");
        if (header.find("'fortran_order': True") != std::string::npos)
            throw std::runtime_error("fortran_order unsupported");
        a.shape = parse_shape(header);
        long long n = a.numel();
        const unsigned char* dp = raw.data() + 10 + hlen;
        if (a.dtype == "<f8" || a.dtype == "=f8" || a.dtype == "f8") {
            a.isfloat = true; a.f.resize(n);
            for (long long k = 0; k < n; ++k)
                std::memcpy(&a.f[k], dp + k * 8, 8);
        } else if (a.dtype == "<i8" || a.dtype == "i8") {
            a.isfloat = false; a.i.resize(n);
            for (long long k = 0; k < n; ++k)
                std::memcpy(&a.i[k], dp + k * 8, 8);
        } else if (a.dtype == "<i4" || a.dtype == "i4") {
            a.isfloat = false; a.i.resize(n);
            for (long long k = 0; k < n; ++k) {
                int v; std::memcpy(&v, dp + k * 4, 4); a.i[k] = v;
            }
        } else if (a.dtype == "|u1" || a.dtype == "u1") {
            a.isfloat = false; a.i.resize(n);
            for (long long k = 0; k < n; ++k) a.i[k] = (long long)dp[k];
        } else {
            throw std::runtime_error("unsupported dtype " + a.dtype);
        }
        return a;
    }

    std::vector<unsigned char> build() const {
        long long n = numel();
        std::vector<unsigned char> data(n * 8);
        if (isfloat) {
            for (long long k = 0; k < n; ++k)
                std::memcpy(data.data() + k * 8, &f[k], 8);
        } else {
            for (long long k = 0; k < n; ++k)
                std::memcpy(data.data() + k * 8, &i[k], 8);
        }
        std::string sh;
        if (shape.size() == 1) {
            sh = "(" + std::to_string(shape[0]) + ",)";
        } else {
            sh = "(";
            for (size_t k = 0; k < shape.size(); ++k) {
                sh += std::to_string(shape[k]);
                if (k + 1 < shape.size()) sh += ", ";
            }
            sh += ")";
        }
        std::string base = "{'descr': '" + dtype +
            "', 'fortran_order': False, 'shape': " + sh + ", }";
        std::string header; unsigned hlen = 0;
        for (int pad = 0; ; ++pad) {
            header = base + std::string(pad, ' ') + "\n";
            if ((10 + header.size()) % 64 == 0) { hlen = (unsigned)header.size(); break; }
        }
        std::vector<unsigned char> out;
        static const unsigned char magic[6] = {0x93,'N','U','M','P','Y'};
        out.insert(out.end(), magic, magic + 6);
        out.push_back(1); out.push_back(0);
        out.push_back(hlen & 0xff); out.push_back((hlen >> 8) & 0xff);
        out.insert(out.end(), header.begin(), header.end());
        out.insert(out.end(), data.begin(), data.end());
        return out;
    }

private:
    static std::string quoted_after(const std::string& s,
                                    const std::string& key) {
        std::string pat = key + "':";
        size_t p = s.find(pat);
        if (p == std::string::npos) { pat = key + ":"; p = s.find(pat); }
        size_t q1 = s.find("'", p + pat.size());
        size_t q2 = s.find("'", q1 + 1);
        return s.substr(q1 + 1, q2 - q1 - 1);
    }
    static std::vector<long long> parse_shape(const std::string& s) {
        size_t p = s.find("'shape':");
        size_t l = s.find("(", p), r = s.find(")", l);
        std::string inside = s.substr(l + 1, r - l - 1);
        std::vector<long long> out; std::string cur;
        for (size_t k = 0; k < inside.size(); ++k) {
            char c = inside[k];
            if (c == ',' || c == ' ') {
                if (!cur.empty()) { out.push_back(atoll(cur.c_str())); cur.clear(); }
            } else cur.push_back(c);
        }
        if (!cur.empty()) out.push_back(atoll(cur.c_str()));
        return out;
    }
};

// ----------------------------------------------------------------------
// zlib raw-deflate / inflate（ZIP 用，windowBits=-15）
// ----------------------------------------------------------------------
inline std::vector<unsigned char>
inflate_raw(const std::vector<unsigned char>& in, unsigned long expected) {
    z_stream strm; std::memset(&strm, 0, sizeof(strm));
    if (inflateInit2(&strm, -15) != Z_OK)
        throw std::runtime_error("inflateInit2 failed");
    std::vector<unsigned char> out(expected);
    strm.next_in = const_cast<Bytef*>(in.data()); strm.avail_in = (uInt)in.size();
    strm.next_out = out.data(); strm.avail_out = (uInt)out.size();
    int rc = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);
    if (rc != Z_STREAM_END) throw std::runtime_error("inflate failed");
    return out;
}
inline std::vector<unsigned char>
deflate_raw(const std::vector<unsigned char>& in) {
    z_stream strm; std::memset(&strm, 0, sizeof(strm));
    if (deflateInit2(&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -15, 8,
                     Z_DEFAULT_STRATEGY) != Z_OK)
        throw std::runtime_error("deflateInit2 failed");
    std::vector<unsigned char> out(deflateBound(&strm, (uLong)in.size()));
    strm.next_in = const_cast<Bytef*>(in.data()); strm.avail_in = (uInt)in.size();
    strm.next_out = out.data(); strm.avail_out = (uInt)out.size();
    deflate(&strm, Z_FINISH);
    out.resize(strm.total_out);
    deflateEnd(&strm);
    return out;
}

inline std::vector<unsigned char> read_file_bytes(const std::string& path) {
    std::ifstream f(path.c_str(), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
    return buf;
}
inline void write_file_bytes(const std::string& path,
                             const std::vector<unsigned char>& data) {
    std::ofstream f(path.c_str(), std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f.write((const char*)data.data(), (std::streamsize)data.size());
}

// ----------------------------------------------------------------------
// NPZ 归档：真正的 ZIP 容器读写
// ----------------------------------------------------------------------
class NpzArchive {
public:
    std::map<std::string, NpyArray> arrays;

    bool has(const std::string& key) const {
        return arrays.find(key) != arrays.end();
    }
    NpyArray& get(const std::string& key) { return arrays[key]; }
    void put(const std::string& key, const NpyArray& a) { arrays[key] = a; }

    static unsigned long rd32(const unsigned char* p) {
        return (unsigned long)p[0] | (p[1] << 8) |
               (p[2] << 16) | ((unsigned long)p[3] << 24);
    }
    static unsigned rd16(const unsigned char* p) {
        return (unsigned)(p[0] | (p[1] << 8));
    }
    static void wr32(std::vector<unsigned char>& v, unsigned long x) {
        v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff);
        v.push_back((x >> 16) & 0xff); v.push_back((x >> 24) & 0xff);
    }
    static void wr16(std::vector<unsigned char>& v, unsigned x) {
        v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff);
    }

    void load(const std::string& path) {
        std::vector<unsigned char> buf = read_file_bytes(path);
        if (buf.size() < 22) throw std::runtime_error("file too small");
        long eocd = -1;
        for (long i = (long)buf.size() - 22; i >= 0; --i) {
            if (rd32(buf.data() + i) == 0x06054b50) { eocd = i; break; }
        }
        if (eocd < 0) throw std::runtime_error("EOCD not found");
        unsigned cdcount = rd16(buf.data() + eocd + 10);
        unsigned long cdoff = rd32(buf.data() + eocd + 16);
        size_t pos = (size_t)cdoff;
        for (unsigned q = 0; q < cdcount; ++q) {
            if (rd32(buf.data() + pos) != 0x02014b50)
                throw std::runtime_error("bad central entry");
            unsigned method = rd16(buf.data() + pos + 10);
            unsigned long csize = rd32(buf.data() + pos + 20);
            unsigned long usize = rd32(buf.data() + pos + 24);
            unsigned fnlen = rd16(buf.data() + pos + 28);
            unsigned exlen = rd16(buf.data() + pos + 30);
            unsigned cmlen = rd16(buf.data() + pos + 32);
            unsigned long lho = rd32(buf.data() + pos + 42);
            std::string name((const char*)buf.data() + pos + 46, fnlen);
            pos += 46 + fnlen + exlen + cmlen;
            unsigned lfn = rd16(buf.data() + lho + 26);
            unsigned lex = rd16(buf.data() + lho + 28);
            size_t dstart = lho + 30 + lfn + lex;
            std::vector<unsigned char> comp(buf.begin() + dstart,
                                           buf.begin() + dstart + csize);
            std::vector<unsigned char> raw =
                (method == 0) ? comp : inflate_raw(comp, usize);
            try {
                NpyArray arr = NpyArray::parse(raw);
                std::string key = name;
                if (key.size() > 4 && key.substr(key.size() - 4) == ".npy")
                    key = key.substr(0, key.size() - 4);
                arrays[key] = arr;
            } catch (const std::exception&) {
                // 跳过不支持的条目（如 <U13 字符串数组 channel_names）
            }
        }
    }

    void save(const std::string& path) const {
        std::vector<unsigned char> out;
        struct Ent {
            std::string name; unsigned long crc, csize, usize, offset;
        };
        std::vector<Ent> ents;
        for (std::map<std::string, NpyArray>::const_iterator it = arrays.begin();
             it != arrays.end(); ++it) {
            std::string fname = it->first + ".npy";
            std::vector<unsigned char> raw = it->second.build();
            std::vector<unsigned char> comp = deflate_raw(raw);
            unsigned long crc = crc32(0L, raw.data(), (uInt)raw.size());
            unsigned long offset = (unsigned long)out.size();
            wr32(out, 0x04034b50); wr16(out, 20); wr16(out, 0);
            wr16(out, 8); wr16(out, 0); wr16(out, 0);
            wr32(out, crc); wr32(out, (unsigned long)comp.size());
            wr32(out, (unsigned long)raw.size());
            wr16(out, (unsigned)fname.size()); wr16(out, 0);
            out.insert(out.end(), fname.begin(), fname.end());
            out.insert(out.end(), comp.begin(), comp.end());
            Ent e; e.name = fname; e.crc = crc;
            e.csize = (unsigned long)comp.size();
            e.usize = (unsigned long)raw.size(); e.offset = offset;
            ents.push_back(e);
        }
        unsigned long cdoff = (unsigned long)out.size();
        for (size_t q = 0; q < ents.size(); ++q) {
            const Ent& e = ents[q];
            // central header 比 local header 多一个 "version made by" 字段
            wr32(out, 0x02014b50);
            wr16(out, 20);   // version made by
            wr16(out, 20);   // version needed
            wr16(out, 0);    // flags
            wr16(out, 8);    // method = deflate
            wr16(out, 0);    // mod time
            wr16(out, 0);    // mod date
            wr32(out, e.crc);
            wr32(out, e.csize);
            wr32(out, e.usize);
            wr16(out, (unsigned)e.name.size());
            wr16(out, 0);    // extra len
            wr16(out, 0);    // comment len
            wr16(out, 0);    // disk number
            wr16(out, 0);    // internal attrs
            wr32(out, 0);    // external attrs
            wr32(out, e.offset);
            out.insert(out.end(), e.name.begin(), e.name.end());
        }
        unsigned long cdsize = (unsigned long)out.size() - cdoff;
        wr32(out, 0x06054b50); wr16(out, 0); wr16(out, 0);
        wr16(out, (unsigned)ents.size()); wr16(out, (unsigned)ents.size());
        wr32(out, cdsize); wr32(out, cdoff); wr16(out, 0);
        write_file_bytes(path, out);
    }
};

// ----------------------------------------------------------------------
// 稠密矩阵（行优先），矩阵乘用线程池并行
// ----------------------------------------------------------------------
class DMat {
public:
    int r, c;
    std::vector<double> a;
    DMat(int r_ = 0, int c_ = 0)
        : r(r_), c(c_), a((size_t)r_ * c_, 0.0) {}
    double& at(int i, int j) { return a[(size_t)i * c + j]; }
    double get(int i, int j) const { return a[(size_t)i * c + j]; }
    void set(int i, int j, double v) { a[(size_t)i * c + j] = v; }

    DMat transpose() const {
        DMat t(c, r);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j) t.set(j, i, get(i, j));
        return t;
    }
    DMat mul(const DMat& B, ThreadPool& pool) const {
        DMat C(r, B.c);
        const DMat& A = *this;
        pool.parallel_for(0, r, std::function<void(int)>([&](int i) {
            for (int k = 0; k < A.c; ++k) {
                double aik = A.get(i, k);
                if (aik == 0.0) continue;
                for (int j = 0; j < B.c; ++j)
                    C.at(i, j) += aik * B.get(k, j);
            }
        }));
        return C;
    }
};

} // namespace qpfc
