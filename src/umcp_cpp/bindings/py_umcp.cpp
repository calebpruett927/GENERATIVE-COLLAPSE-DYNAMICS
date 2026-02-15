/**
 * @file py_umcp.cpp
 * @brief pybind11 Python bindings for the UMCP C++ accelerator.
 *
 * Exposes the C++ kernel, seam, and integrity modules to Python
 * with NumPy array support for zero-copy data transfer.
 *
 * Usage from Python:
 *   import umcp_accel
 *   result = umcp_accel.compute_kernel(c_array, w_array)
 *   results = umcp_accel.compute_kernel_batch(trace_2d, w_array)
 *   sha = umcp_accel.hash_file("path/to/file")
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "umcp/kernel.hpp"
#include "umcp/seam.hpp"
#include "umcp/integrity.hpp"

namespace py = pybind11;

// ─── Kernel bindings ────────────────────────────────────────────────

/**
 * Compute kernel invariants from NumPy arrays.
 * Zero-copy: reads directly from NumPy buffer.
 */
py::dict compute_kernel_py(py::array_t<double> c_arr,
                           py::array_t<double> w_arr,
                           double epsilon,
                           bool validate) {
    auto c = c_arr.unchecked<1>();
    auto w = w_arr.unchecked<1>();

    if (c.shape(0) != w.shape(0)) {
        throw std::invalid_argument("c and w must have same length");
    }

    std::size_t n = static_cast<std::size_t>(c.shape(0));

    umcp::KernelComputer computer(epsilon);
    auto result = computer.compute(c.data(0), w.data(0), n, validate);

    py::dict out;
    out["F"] = result.F;
    out["omega"] = result.omega;
    out["S"] = result.S;
    out["C"] = result.C;
    out["kappa"] = result.kappa;
    out["IC"] = result.IC;
    out["amgm_gap"] = result.delta;
    out["regime"] = result.regime;
    out["is_homogeneous"] = result.is_homogeneous;
    out["computation_mode"] = result.computation_mode;
    return out;
}

/**
 * Batch compute kernel invariants over a trace matrix (T × n).
 * Returns a dict of NumPy arrays for vectorized downstream use.
 */
py::dict compute_kernel_batch_py(py::array_t<double> trace_arr,
                                 py::array_t<double> w_arr,
                                 double epsilon) {
    auto trace = trace_arr.unchecked<2>();
    auto w = w_arr.unchecked<1>();

    std::size_t T = static_cast<std::size_t>(trace.shape(0));
    std::size_t n = static_cast<std::size_t>(trace.shape(1));

    if (static_cast<std::size_t>(w.shape(0)) != n) {
        throw std::invalid_argument("Weight dim must match trace columns");
    }

    umcp::KernelComputer computer(epsilon);
    auto results = computer.compute_batch(trace.data(0, 0), w.data(0), T, n);

    // Pack into NumPy arrays
    py::array_t<double> F_out(T), omega_out(T), S_out(T), C_out(T);
    py::array_t<double> kappa_out(T), IC_out(T), delta_out(T);

    auto F_buf = F_out.mutable_unchecked<1>();
    auto omega_buf = omega_out.mutable_unchecked<1>();
    auto S_buf = S_out.mutable_unchecked<1>();
    auto C_buf = C_out.mutable_unchecked<1>();
    auto kappa_buf = kappa_out.mutable_unchecked<1>();
    auto IC_buf = IC_out.mutable_unchecked<1>();
    auto delta_buf = delta_out.mutable_unchecked<1>();

    for (std::size_t t = 0; t < T; ++t) {
        F_buf(t) = results[t].F;
        omega_buf(t) = results[t].omega;
        S_buf(t) = results[t].S;
        C_buf(t) = results[t].C;
        kappa_buf(t) = results[t].kappa;
        IC_buf(t) = results[t].IC;
        delta_buf(t) = results[t].delta;
    }

    py::dict out;
    out["F"] = F_out;
    out["omega"] = omega_out;
    out["S"] = S_out;
    out["C"] = C_out;
    out["kappa"] = kappa_out;
    out["IC"] = IC_out;
    out["delta"] = delta_out;
    return out;
}

/**
 * Classify regime from kernel outputs.
 */
std::string classify_regime_py(double omega, double F, double S, double C,
                               double omega_thresh, double omega_watch) {
    umcp::KernelOutputs mock{};
    mock.omega = omega;
    mock.F = F;
    mock.S = S;
    mock.C = C;

    umcp::RegimeThresholds thresh{};
    thresh.omega_collapse = omega_thresh;
    thresh.omega_watch = omega_watch;

    return umcp::KernelComputer::classify_regime(mock, thresh);
}


// ─── Seam bindings ──────────────────────────────────────────────────

class PySeamChain {
public:
    explicit PySeamChain(double alpha = 0.05, std::size_t K_max = 1000)
        : acc_(alpha, K_max) {}

    py::dict add_seam(int t0, int t1,
                      double kappa_t0, double kappa_t1,
                      double tau_R,
                      double R = 0.01,
                      double D_omega = 0.0,
                      double D_C = 0.0) {
        auto record = acc_.add_seam(t0, t1, kappa_t0, kappa_t1,
                                    tau_R, R, D_omega, D_C);
        py::dict out;
        out["t0"] = record.t0;
        out["t1"] = record.t1;
        out["delta_kappa_ledger"] = record.delta_kappa_ledger;
        out["delta_kappa_budget"] = record.delta_kappa_budget;
        out["residual"] = record.residual;
        out["cumulative_residual"] = record.cumulative_residual;
        return out;
    }

    double total_delta_kappa() const { return acc_.total_delta_kappa(); }
    std::size_t size() const { return acc_.size(); }
    bool failure_detected() const { return acc_.failure_detected(); }

    py::dict get_metrics() const {
        auto m = acc_.get_metrics();
        py::dict out;
        out["total_seams"] = m.total_seams;
        out["total_delta_kappa"] = m.total_delta_kappa;
        out["cumulative_abs_residual"] = m.cumulative_abs_residual;
        out["max_residual"] = m.max_residual;
        out["mean_residual"] = m.mean_residual;
        out["growth_exponent"] = m.growth_exponent;
        out["is_returning"] = m.is_returning;
        out["failure_detected"] = m.failure_detected;
        return out;
    }

private:
    umcp::SeamChainAccumulator acc_;
};


// ─── Module definition ──────────────────────────────────────────────

PYBIND11_MODULE(umcp_accel, m) {
    m.doc() = R"doc(
        UMCP C++ Accelerator

        High-performance implementations of kernel computation, seam chain
        accumulation, and SHA-256 integrity. Drop-in replacement for the
        Python implementations with identical numerical results.

        Tier-0 Protocol: no Tier-1 symbol is redefined.
        All frozen parameters come from the Python-side contract.
    )doc";

    // ── Kernel ──
    m.def("compute_kernel", &compute_kernel_py,
          py::arg("c"), py::arg("w"),
          py::arg("epsilon") = 1e-8,
          py::arg("validate") = true,
          R"doc(
              Compute kernel invariants (F, ω, S, C, κ, IC) from trace vector.

              Parameters:
                  c: numpy array of coordinates, c ∈ [ε, 1−ε]^n
                  w: numpy array of weights, sum(w) = 1
                  epsilon: Guard band (frozen, default 10⁻⁸)
                  validate: Whether to validate output ranges (Lemma 1)

              Returns:
                  dict with keys: F, omega, S, C, kappa, IC, amgm_gap, regime
          )doc");

    m.def("compute_kernel_batch", &compute_kernel_batch_py,
          py::arg("trace"), py::arg("w"),
          py::arg("epsilon") = 1e-8,
          R"doc(
              Batch compute kernel invariants over a trace matrix.

              Parameters:
                  trace: numpy array (T × n) of coordinates
                  w: numpy array (n,) of weights
                  epsilon: Guard band

              Returns:
                  dict of numpy arrays: F, omega, S, C, kappa, IC, delta
          )doc");

    m.def("classify_regime", &classify_regime_py,
          py::arg("omega"), py::arg("F"),
          py::arg("S"), py::arg("C"),
          py::arg("omega_collapse") = 0.30,
          py::arg("omega_watch") = 0.15,
          "Classify regime from kernel outputs.");

    // ── Error propagation ──
    m.def("propagate_error", [](double delta_c, double epsilon) {
        umcp::KernelComputer computer(epsilon);
        auto bounds = computer.propagate_error(delta_c);
        py::dict out;
        out["F"] = bounds.F;
        out["omega"] = bounds.omega;
        out["kappa"] = bounds.kappa;
        out["S"] = bounds.S;
        return out;
    }, py::arg("delta_c"), py::arg("epsilon") = 1e-8,
    "OPT-12: Lipschitz error propagation (Lemma 23).");

    // ── Seam chain ──
    py::class_<PySeamChain>(m, "SeamChain",
        "High-performance seam chain accumulator (OPT-10, OPT-11).")
        .def(py::init<double, std::size_t>(),
             py::arg("alpha") = 0.05, py::arg("K_max") = 1000)
        .def("add_seam", &PySeamChain::add_seam,
             py::arg("t0"), py::arg("t1"),
             py::arg("kappa_t0"), py::arg("kappa_t1"),
             py::arg("tau_R"),
             py::arg("R") = 0.01,
             py::arg("D_omega") = 0.0,
             py::arg("D_C") = 0.0,
             "Add a seam with O(1) incremental update.")
        .def("total_delta_kappa", &PySeamChain::total_delta_kappa,
             "O(1) query for total ledger change.")
        .def("size", &PySeamChain::size)
        .def("failure_detected", &PySeamChain::failure_detected)
        .def("get_metrics", &PySeamChain::get_metrics,
             "Compute comprehensive seam chain metrics.");

    // ── SHA-256 integrity ──
    m.def("hash_file", &umcp::SHA256Hasher::hash_file,
          py::arg("filepath"),
          "Compute SHA-256 hash of a file (hex-encoded, 64 chars).");

    m.def("hash_bytes", [](py::bytes data) {
        std::string s = data;
        return umcp::SHA256Hasher::hash_bytes(s.data(), s.size());
    }, py::arg("data"),
    "Compute SHA-256 hash of bytes.");

    m.def("verify_file", &umcp::SHA256Hasher::verify_file,
          py::arg("filepath"), py::arg("expected_hash"),
          "Verify a file against expected SHA-256 hash.");

    m.def("hash_files", &umcp::SHA256Hasher::hash_files,
          py::arg("filepaths"),
          "Batch hash multiple files.");

    // ── Version info ──
    m.attr("__version__") = "1.0.0";
    m.attr("__tier__") = "Tier-0 (Protocol)";
}
