// 10_gnn_gcn_class.cpp —— 两层 GCNConv 编码器（类实现版，公式 4/5）
// ============================================================================
// 与 Python 10_gnn_gcn.py 数值对齐，但采用类封装（非函数式）：
//   · 直接加载/保存真正的 .npz（NpzArchive，ZIP+raw-deflate，与 numpy 互通）
//   · 矩阵乘 / ReLU / 边置信度用 ThreadPool 线程池并行
// 编译见 build_class.bat（g++ -std=c++11，链接 zlib）。
#include "npz_archive.hpp"

#ifdef _WIN32
#include <direct.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using namespace qpfc;

static const int H_DIM = 16;
static const int EPOCHS = 300;
static const double LR = 1e-2;
static const long long GCN_SEED = 20261014LL + 11;

static double sigmoid_d(double x) {
    if (x < -60) x = -60; if (x > 60) x = 60;
    return 1.0 / (1.0 + std::exp(-x));
}

// 从共享 CSV 加载 rows×cols 矩阵（逗号分隔，%.17g）
static bool load_csv(const std::string& path, DMat& M) {
    std::ifstream f(path.c_str());
    if (!f) return false;
    std::string line; int rr = 0;
    while (rr < M.r && std::getline(f, line)) {
        int cc = 0; size_t st = 0;
        for (size_t i = 0; i <= line.size() && cc < M.c; ++i) {
            if (i == line.size() || line[i] == ',') {
                M.set(rr, cc, std::strtod(line.substr(st, i - st).c_str(), NULL));
                ++cc; st = i + 1;
            }
        }
        ++rr;
    }
    return rr == M.r;
}

// ======================================================================
// GCNModel：两层 GCN 自监督模型（封装参数、图、Adam 状态与训练过程）
// ======================================================================
class GCNModel {
public:
    GCNModel(int n, int hdim, ThreadPool& pool)
        : n_(n), h_(hdim), E_(0), pool_(pool),
          X_(n, 2), P_(n, n),
          W0_(2, hdim), W1_(hdim, hdim),
          b0_(hdim, 0.0), b1_(hdim, 0.0),
          mW0_(2, hdim), vW0_(2, hdim),
          mW1_(hdim, hdim), vW1_(hdim, hdim),
          mb0_(hdim, 0.0), vb0_(hdim, 0.0),
          mb1_(hdim, 0.0), vb1_(hdim, 0.0) {}

    // 装载归一化后的节点特征、邻接与边监督目标
    void set_inputs(const DMat& X, const DMat& A,
                    const std::vector<int>& eu,
                    const std::vector<int>& ev,
                    const std::vector<double>& tw) {
        X_ = X; eu_ = eu; ev_ = ev; tw_ = tw; E_ = (int)eu.size();
        P_ = build_propagation(A, pool_);
    }

    // 初始化权重：优先共享 CSV，缺失则回退 RNG
    void init_weights(const std::string& w0csv, const std::string& w1csv) {
        bool ok = load_csv(w0csv, W0_) && load_csv(w1csv, W1_);
        if (ok) return;
        std::cerr << "[警告] 未找到 " << w0csv << " / " << w1csv
                  << "，回退 mt19937 初始化（与 Python 不一致）。\n";
        std::mt19937 eng((unsigned)GCN_SEED);
        std::normal_distribution<double> n01(0.0, 1.0);
        for (int q = 0; q < 2 * h_; ++q) W0_.a[q] = n01(eng) * 0.5;
        for (int q = 0; q < h_ * h_; ++q) W1_.a[q] = n01(eng) * 0.3;
    }

    // P = Dhat^-1/2 (A+I) Dhat^-1/2（按行并行计算归一化）
    static DMat build_propagation(const DMat& A, ThreadPool& pool) {
        int n = A.r;
        DMat Ahat(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Ahat.set(i, j, A.get(i, j) + (i == j ? 1.0 : 0.0));
        std::vector<double> dinv(n);
        pool.parallel_for(0, n, std::function<void(int)>([&](int i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) s += Ahat.get(i, j);
            dinv[i] = s > 0 ? 1.0 / std::sqrt(s) : 0.0;
        }));
        DMat P(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                P.set(i, j, dinv[i] * Ahat.get(i, j) * dinv[j]);
        return P;
    }

    // 前向：返回 Z，同时输出 M1/H1 供反向使用
    DMat forward(DMat& M1, DMat& H1) {
        DMat PX = P_.mul(X_, pool_);
        M1 = PX.mul(W0_, pool_); add_bias(M1, b0_);
        H1 = DMat(n_, h_);
        pool_.parallel_for(0, n_, std::function<void(int)>([&](int i) {
            for (int c = 0; c < h_; ++c)
                H1.set(i, c, std::max(0.0, M1.get(i, c)));
        }));
        DMat PH1 = P_.mul(H1, pool_);
        DMat Z = PH1.mul(W1_, pool_); add_bias(Z, b1_);
        return Z;
    }

    // 边置信度 phi = sigmoid(z_u . z_v)，按边并行
    std::vector<double> edge_phi(const DMat& Z,
                                 std::vector<double>& ginner) {
        std::vector<double> phi(E_);
        ginner.assign(E_, 0.0);
        pool_.parallel_for(0, E_, std::function<void(int)>([&](int e) {
            double inner = 0.0;
            for (int c = 0; c < h_; ++c)
                inner += Z.get(eu_[e], c) * Z.get(ev_[e], c);
            double p = sigmoid_d(inner);
            phi[e] = p;
            double gphi = 2.0 * (p - tw_[e]) / E_;
            ginner[e] = gphi * p * (1.0 - p);
        }));
        return phi;
    }

    // 完整训练循环
    void train(int epochs) {
        const double b1m = 0.9, b2m = 0.999, eps = 1e-8;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            DMat M1, H1;
            DMat Z = forward(M1, H1);
            std::vector<double> ginner;
            std::vector<double> phi = edge_phi(Z, ginner);

            // dZ：每条边两端累加（串行，避免同节点写冲突）
            DMat dZ(n_, h_);
            for (int e = 0; e < E_; ++e) {
                for (int c = 0; c < h_; ++c) {
                    dZ.at(eu_[e], c) += ginner[e] * Z.get(ev_[e], c);
                    dZ.at(ev_[e], c) += ginner[e] * Z.get(eu_[e], c);
                }
            }
            // 反向第二层
            DMat PdZ = P_.mul(dZ, pool_);
            DMat dW1 = H1.transpose().mul(PdZ, pool_);
            std::vector<double> db1(h_, 0.0);
            for (int i = 0; i < n_; ++i)
                for (int c = 0; c < h_; ++c) db1[c] += dZ.get(i, c);
            DMat dH1 = PdZ.mul(W1_.transpose(), pool_);
            DMat dM1(n_, h_);
            for (int i = 0; i < n_; ++i)
                for (int c = 0; c < h_; ++c)
                    dM1.set(i, c, M1.get(i, c) > 0 ? dH1.get(i, c) : 0.0);
            // 反向第一层
            DMat PdM1 = P_.mul(dM1, pool_);
            DMat dW0 = X_.transpose().mul(PdM1, pool_);
            std::vector<double> db0(h_, 0.0);
            for (int i = 0; i < n_; ++i)
                for (int c = 0; c < h_; ++c) db0[c] += dM1.get(i, c);

            double loss = 0.0;
            for (int e = 0; e < E_; ++e)
                loss += (phi[e] - tw_[e]) * (phi[e] - tw_[e]);
            loss /= E_;

            // Adam 更新（bias 修正）
            double bc1 = 1.0 - std::pow(b1m, epoch + 1);
            double bc2 = 1.0 - std::pow(b2m, epoch + 1);
            adam_mat(W0_, mW0_, vW0_, dW0, b1m, b2m, eps, bc1, bc2);
            adam_mat(W1_, mW1_, vW1_, dW1, b1m, b2m, eps, bc1, bc2);
            adam_vec(b0_, mb0_, vb0_, db0, b1m, b2m, eps, bc1, bc2);
            adam_vec(b1_, mb1_, vb1_, db1, b1m, b2m, eps, bc1, bc2);

            if (epoch % 50 == 0 || epoch == epochs - 1) {
                std::printf("  epoch %4d  MSE=%.5f\n", epoch, loss);
                std::fflush(stdout);
            }
        }
    }

    // 训练后再前向一次，返回 Z 与 phi
    DMat finalize(std::vector<double>& phi) {
        DMat M1, H1;
        DMat Z = forward(M1, H1);
        std::vector<double> ginner;
        phi = edge_phi(Z, ginner);
        return Z;
    }

    long long param_count() const {
        return (long long)W0_.a.size() + (long long)b0_.size()
             + (long long)W1_.a.size() + (long long)b1_.size();
    }
    const DMat& X() const { return X_; }
    const DMat& P() const { return P_; }

private:
    int n_, h_, E_;
    ThreadPool& pool_;
    DMat X_, P_, W0_, W1_;
    std::vector<double> b0_, b1_;
    DMat mW0_, vW0_, mW1_, vW1_;
    std::vector<double> mb0_, vb0_, mb1_, vb1_;
    std::vector<int> eu_, ev_;
    std::vector<double> tw_;

    static void add_bias(DMat& M, const std::vector<double>& b) {
        for (int i = 0; i < M.r; ++i)
            for (int c = 0; c < M.c; ++c) M.at(i, c) += b[c];
    }
    static void adam_mat(DMat& W, DMat& mw, DMat& vw, const DMat& g,
                         double b1m, double b2m, double eps,
                         double bc1, double bc2) {
        for (size_t q = 0; q < W.a.size(); ++q) {
            mw.a[q] = b1m * mw.a[q] + (1 - b1m) * g.a[q];
            vw.a[q] = b2m * vw.a[q] + (1 - b2m) * g.a[q] * g.a[q];
            double mh = mw.a[q] / bc1, vh = vw.a[q] / bc2;
            W.a[q] -= LR * mh / (std::sqrt(vh) + eps);
        }
    }
    static void adam_vec(std::vector<double>& w, std::vector<double>& mw,
                         std::vector<double>& vw,
                         const std::vector<double>& g,
                         double b1m, double b2m, double eps,
                         double bc1, double bc2) {
        for (size_t q = 0; q < w.size(); ++q) {
            mw[q] = b1m * mw[q] + (1 - b1m) * g[q];
            vw[q] = b2m * vw[q] + (1 - b2m) * g[q] * g[q];
            double mh = mw[q] / bc1, vh = vw[q] / bc2;
            w[q] -= LR * mh / (std::sqrt(vh) + eps);
        }
    }
};

// 确保 cpp/outputs 目录存在
static void ensure_outputs() {
#ifdef _WIN32
    _mkdir("outputs");
#else
    mkdir("outputs", 0755);
#endif
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif
    ensure_outputs();

    // ---------- 加载 Python 写出的真正 .npz ----------
    const std::string in_path = "../outputs/09_graph.npz";
    NpzArchive in;
    std::cout << "加载 NPZ：" << in_path << "\n" << std::flush;
    in.load(in_path);

    const NpyArray& Xa = in.get("X");
    const NpyArray& Aa = in.get("A");
    const NpyArray& ea = in.get("edges");
    int n = (int)Xa.shape[0], E = (int)ea.shape[0];
    std::cout << "节点 " << n << "，边 " << E << "\n";

    // 节点特征归一化：度除以最大度
    DMat X(n, 2), A(n, n);
    double maxdeg = 1.0;
    for (int i = 0; i < n; ++i) maxdeg = std::max(maxdeg, Xa.f[(size_t)i * 2]);
    for (int i = 0; i < n; ++i) {
        X.set(i, 0, Xa.f[(size_t)i * 2] / maxdeg);
        X.set(i, 1, Xa.f[(size_t)i * 2 + 1]);
    }
    A.a = Aa.f;

    // 边端点与软目标 m（edges 第 0/1 列端点，第 5 列 m）
    std::vector<int> eu(E), ev(E);
    std::vector<double> tw(E);
    for (int e = 0; e < E; ++e) {
        eu[e] = (int)ea.f[(size_t)e * 6 + 0];
        ev[e] = (int)ea.f[(size_t)e * 6 + 1];
        tw[e] = ea.f[(size_t)e * 6 + 5];
    }

    // ---------- 线程池 + 模型 ----------
    int nthreads = (int)std::max(2u, std::thread::hardware_concurrency());
#ifdef FORCE_NTHREADS
    nthreads = FORCE_NTHREADS;
#endif
    ThreadPool pool(nthreads);
    std::cout << "线程池工作线程数：" << pool.size() << "\n";

    std::cout << "\n================ 10 两层 GCNConv（类实现）================\n";
    GCNModel model(n, H_DIM, pool);
    model.set_inputs(X, A, eu, ev, tw);
    model.init_weights("../gcn_W0_init.csv", "../gcn_W1_init.csv");

    std::cout << "参数量 = " << model.param_count() << "（应为 320）\n";
    model.train(EPOCHS);

    std::vector<double> phi;
    DMat Z = model.finalize(phi);

    double sse = 0.0;
    for (int e = 0; e < E; ++e) sse += (phi[e] - tw[e]) * (phi[e] - tw[e]);
    std::printf("\n最终 RMSE(phi, w) = %.4f\n", std::sqrt(sse / E));
    std::cout << "边权 w（软目标）vs GNN 置信度 phi（抽 10 条）：\n";
    for (int e = 0; e < std::min(10, E); ++e)
        std::printf("  (%3d,%3d) %8.3f %9.3f\n", eu[e], ev[e], tw[e], phi[e]);

    // ---------- 保存真正的 .npz（cpp/outputs，不覆盖 Python 产物）----------
    NpzArchive out;
    out.put("Z", NpyArray::from_f64(std::vector<long long>{n, H_DIM}, Z.a));
    out.put("phi", NpyArray::from_f64(std::vector<long long>{E}, phi));
    out.put("edges", NpyArray::from_f64(ea.shape, ea.f));
    out.put("edge_type", NpyArray::from_i64(in.get("edge_type").shape,
                                            in.get("edge_type").i));
    out.put("X", NpyArray::from_f64(std::vector<long long>{n, 2}, X.a));
    out.put("A", NpyArray::from_f64(std::vector<long long>{n, n}, A.a));
    out.put("theta", NpyArray::from_f64(in.get("theta").shape,
                                        in.get("theta").f));
    std::vector<std::string> segkeys;
    const char* sk[4] = {"seg_label","seg_channel","seg_conn_id","seg_index"};
    for (int k = 0; k < 4; ++k) {
        const NpyArray& s = in.get(sk[k]);
        out.put(sk[k], NpyArray::from_i64(s.shape, s.i));
    }
    const std::string out_path = "outputs/10_gnn.npz";
    out.save(out_path);
    std::cout << "已保存 NPZ：" << out_path << "\n";
    return 0;
}
