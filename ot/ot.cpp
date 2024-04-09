#include "ot.h"
#include "EMD.h"

namespace ot {
  std::string check_result(int result_code) {
    std::string message;

    if (result_code == ProblemType::OPTIMAL)
      message = "Optimal status has been reached";
    if (result_code == ProblemType::INFEASIBLE)
      message = "Problem infeasible. Check that a and b are in the simplex";
    if (result_code == ProblemType::UNBOUNDED)
      message = "Problem unbounded";
    if (result_code == ProblemType::MAX_ITER_REACHED)
      message =
          "numItermax reached before optimality. Try to increase numItermax";
    return message;
  }

  EMDCluster emd_c(Eigen::ArrayXd a, Eigen::ArrayXd b, RowMajorMatrixXd M,
                   uint64_t max_iter, int numThreads) {
    /**
     Solves the Earth Movers distance problem and returns the optimal transport
    matrix

        gamm=emd(a,b,M)

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the metric cost matrix
    - a and b are the sample weights

    .. warning::
        Note that the M matrix needs to be a C-order :py.cls:`numpy.array`

    .. warning::
        The C++ solver discards all samples in the distributions with
        zeros weights. This means that while the primal variable (transport
        matrix) is exact, the solver only returns feasible dual potentials
        on the samples with weights different from zero.

    Parameters
    ----------
    a : (ns,) numpy.ndarray, float64
        source histogram
    b : (nt,) numpy.ndarray, float64
        target histogram
    M : (ns,nt) numpy.ndarray, float64
        loss matrix
    max_iter : uint64_t
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.

    Returns
    -------
    gamma: (ns x nt) numpy.ndarray
        Optimal transportation matrix for the given parameters

    */
    int n1 = M.rows();
    int n2 = M.cols();
    int nmax = n1 + n2 - 1;
    int result_code_ = 0;
    int nG = 0;

    double cost_ = 0;
    Eigen::ArrayXd alpha_ = Eigen::ArrayXd::Zero(n1);
    Eigen::ArrayXd beta_ = Eigen::ArrayXd::Zero(n2);

    if (a.size() == 0) {
      a = Eigen::ArrayXd::Ones(n1) / n1;
    }

    if (b.size() == 0) {
      b = Eigen::ArrayXd::Ones(n2) / n2;
    }

    RowMajorMatrixXd G_ = RowMajorMatrixXd::Zero(n1, n2);

    if (numThreads == 1) {
      result_code_ = EMD_wrap(n1, n2, a.data(), b.data(), M.data(), G_.data(),
                              alpha_.data(), beta_.data(), &cost_, max_iter);
    } else {
      result_code_ = EMD_wrap_omp(n1, n2, a.data(), b.data(), M.data(),
                                  G_.data(), alpha_.data(), beta_.data(),
                                  &cost_, max_iter, numThreads);
    }

    EMDCluster dataset;
    dataset.alpha = alpha_;
    dataset.beta = beta_;
    dataset.G = G_;
    dataset.cost = cost_;
    dataset.result_code = result_code_;

    return dataset;
  }

  EMDCluster1d emd_1d_sorted(Eigen::ArrayXd u_weights, Eigen::ArrayXd v_weights,
                             Eigen::ArrayXd u, Eigen::ArrayXd v,
                             std::string metric, double p) {
    /**
    Solves the Earth Movers distance problem between sorted 1d measures and
    returns the OT matrix and the associated cost

    Parameters
    ----------
    u_weights : (ns,) ndarray, float64
        Source histogram
    v_weights : (nt,) ndarray, float64
        Target histogram
    u : (ns,) ndarray, float64
        Source dirac locations (on the real line)
    v : (nt,) ndarray, float64
        Target dirac locations (on the real line)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only strings listed in :func:`ot.dist` are accepted.
        Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'` metrics
        are used.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'

    Returns
    -------
    gamma: (n, ) ndarray, float64
        Values in the Optimal transportation matrix
    indices: (n, 2) ndarray, int64
        Indices of the values stored in gamma for the Optimal transportation
        matrix
    cost
        cost associated to the optimal transportation
    */

    double cost_ = 0.f;
    auto n = u_weights.size();
    auto m = v_weights.size();

    int i = 0;
    double w_i = u_weights(0);
    int j = 0;
    double w_j = v_weights(0);

    double m_ij = 0;

    Eigen::ArrayXd G_ = Eigen::ArrayXd::Zero(n + m - 1);
    RowMajorMatrixXi indices_ = RowMajorMatrixXi::Zero(n + m - 1, 2);
    int cur_idx = 0;

    while (true) {
      if (metric == "sqeuclidean") {
        m_ij = std::pow(u(i) - v(j), 2);
      } else if (metric == "cityblock" || metric == "euclidean") {
        m_ij = std::fabs(u(i) - v(j));
      } else if (metric == "minkowski") {
        m_ij = std::pow(std::fabs(u(i) - v(j)), p);
      } else {
        // For custom metrics, you'd likely still handle them outside of
        // ArrayXd's functionality m_ij = customDistanceFunction(u(i), v(j),
        // metric, p);
      }

      if (w_i < w_j || j == m - 1) {
        cost_ += m_ij * w_i;
        G_(cur_idx) = w_i;
        indices_(cur_idx, 0) = i;
        indices_(cur_idx, 1) = j;
        if (++i == n)
          break;
        w_j -= w_i;
        w_i = u_weights(i);
      } else {
        cost_ += m_ij * w_j;
        G_(cur_idx) = w_j;
        indices_(cur_idx, 0) = i;
        indices_(cur_idx, 1) = j;
        if (++j == m)
          break;
        w_i -= w_j;
        w_j = v_weights(j);
      }
      cur_idx++;
    }

    // Adjust the size of G and indices if they are oversized
    G_.conservativeResize(cur_idx);
    indices_.conservativeResize(cur_idx, Eigen::NoChange);

    EMDCluster1d dataset;
    dataset.cost = cost_;
    dataset.G = G_;
    dataset.indices = indices_;

    return dataset;
  }

} // namespace ot