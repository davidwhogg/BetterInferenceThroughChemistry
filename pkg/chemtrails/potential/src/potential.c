#include <math.h>

double sech2_energy(double t, double *pars, double *q, int n_dim) {
    /*  pars:
            G : Gravitational constant
            rho0 : midplane mass density
            z0 : scale height
    */
    double A = 16 * pars[0] * M_PI * pars[1] * pars[2]*pars[2];
    return A * log(cosh(0.5 * q[0] / pars[2]));
}

void sech2_gradient(double t, double *pars, double *q, int n_dim,
                    double *grad) {
    /*  pars:
        G : Gravitational constant
        rho0 : midplane mass density
        z0 : scale height
    */
    double A = 8 * pars[0] * M_PI * pars[1] * pars[2];
    grad[0] = A * tanh(0.5 * q[0] / pars[2]);
}
