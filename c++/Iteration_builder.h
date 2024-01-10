#include <Eigen/Dense>

class Params
{
    public:
    size_t p;
    Eigen::VectorXd gamma, beta, Gamma;
    Params(
        const Eigen::VectorXd & gamma_,
        const Eigen::VectorXd & beta_
    );
};

class IterBuilder
{
    public:
    IterBuilder(const size_t & p):_p(p){}
    double Evaluate(const Params & params);
    Eigen::VectorXd Gradient(
        const Params & params, 
        const double & delta
    );

    double operator()(
        const Eigen::VectorXd &x,
        Eigen::VectorXd & grad
    );

    private:
    size_t _p;
};