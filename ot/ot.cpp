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

  void center_ot_dual(Eigen::ArrayXd& alpha0, Eigen::ArrayXd& beta0,
                      Eigen::ArrayXd& a, Eigen::ArrayXd& b) {
    /**
    The main idea of this function is to find unique dual potentials
    that ensure some kind of centering/fairness. The main idea is to find dual
    potentials that lead to the same final objective value for both source and
    targets (see below for more details). It will help having stability when
    multiple calling of the OT solver with small changes.

    Basically we add another constraint to the potential that will not
    change the objective value but will ensure unicity. The constraint
    is the following:

    .. math::
        \alpha^T \mathbf{a} = \beta^T \mathbf{b}

    in addition to the OT problem constraints.

    since :math:`\sum_i a_i=\sum_j b_j` this can be solved by adding/removing
    a constant from both  :math:`\alpha_0` and :math:`\beta_0`.

    .. math::
        c &= \frac{\beta_0^T \mathbf{b} - \alpha_0^T \mathbf{a}}{\mathbf{1}^T
    \mathbf{b} + \mathbf{1}^T \mathbf{a}}

        \alpha &= \alpha_0 + c

        \beta &= \beta_0 + c
    */
    if (a.size() == 0) {
      a = Eigen::ArrayXd::Ones(alpha0.size()) / alpha0.size();
    }

    if (b.size() == 0) {
      b = Eigen::ArrayXd::Ones(beta0.size()) / beta0.size();
    }

    auto c =
        (b.matrix().dot(beta0.matrix()) - a.matrix().dot(alpha0.matrix())) /
        (a.sum() + b.sum());

    alpha0 += c;
    beta0 -= c;
  }

  void estimate_dual_null_weights(Eigen::ArrayXd& alpha0, Eigen::ArrayXd& beta0,
                                  const Eigen::ArrayXd& a,
                                  const Eigen::ArrayXd& b,
                                  const RowMajorMatrixXd& M) {
    /**Estimate feasible values for 0-weighted dual potentials

    The feasible values are computed efficiently but rather coarsely.

    .. warning::
        This function is necessary because the C++ solver in `emd_c`
        discards all samples in the distributions with
        zeros weights. This means that while the primal variable (transport
        matrix) is exact, the solver only returns feasible dual potentials
        on the samples with weights different from zero.

    First we compute the constraints violations:

    .. math::
        \mathbf{V} = \alpha + \beta^T - \mathbf{M}

    Next we compute the max amount of violation per row (:math:`\alpha`) and
    columns (:math:`beta`)

    .. math::
        \mathbf{v^a}_i = \max_j \mathbf{V}_{i,j}

        \mathbf{v^b}_j = \max_i \mathbf{V}_{i,j}

    Finally we update the dual potential with 0 weights if a
    constraint is violated

    .. math::
        \alpha_i = \alpha_i - \mathbf{v^a}_i \quad \text{ if } \mathbf{a}_i=0
    \text{ and } \mathbf{v^a}_i>0

        \beta_j = \beta_j - \mathbf{v^b}_j \quad \text{ if } \mathbf{b}_j=0
    \text{ and } \mathbf{v^b}_j > 0

    In the end the dual potentials are centered using function
    :py:func:`ot.lp.center_ot_dual`.

    Note that all those updates do not change the objective value of the
    solution but provide dual potentials that do not violate the constraints.
     */
  }

  RowMajorMatrixXd emd(Eigen::ArrayXd& a, Eigen::ArrayXd& b,
                       const RowMajorMatrixXd& M, uint64_t numIterMax,
                       int numThreads, bool center_dual) {
    b = b * a.sum() / b.sum();

    EMDCluster crater;
    crater = emd_c(a, b, M, numIterMax, numThreads);

    if (center_dual) {
      center_ot_dual(crater.alpha, crater.beta, a, b);
    }

    return crater.G;
  }

} // namespace ot