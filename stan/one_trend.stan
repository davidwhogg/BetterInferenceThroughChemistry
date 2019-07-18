functions{
    real sech2_potential(real z, real G, real Sigma, real hz) {
        return 4*pi()*G * Sigma * hz * log(cosh(0.5 * z / hz));
    }
}

data {
    int n_stars;
    vector[n_stars] z;
    vector[n_stars] v_z;
    vector[n_stars] elem;

    // int mean_poly_deg;
    real G;
}

transformed data {

}

parameters {
    real<lower=-100, upper=100> sun_z; // pc
    real<lower=-100, upper=100> sun_vz; // pc/Myr
    real<lower=0, upper=8> ln_Sigma; // Msun/pc^2
    real<lower=0, upper=8> ln_hz; // log(pc)

    real<lower=-7, upper=1> ln_s;
    real<lower=-3, upper=3> inv0;
    real<lower=-1, upper=1> a0;
    real<lower=-1, upper=1> a1;
    real<lower=-1, upper=1> a2;
}

transformed parameters {
    real s;
    real hz;
    real Sigma;
    vector[n_stars] Ez;
    vector[n_stars] invariant;
    vector[n_stars] mu;

    s = exp(ln_s);
    hz = exp(ln_hz);
    Sigma = exp(ln_Sigma);

    for (n in 1:n_stars) {
        Ez[n] = 0.5 * square(v_z[n]) + sech2_potential(z[n], G, Sigma, hz);
    }
    invariant = log(Ez) - mean(log(Ez));

    for (n in 1:n_stars) {
        mu[n] = a0 +
            a1 * (invariant[n] - inv0) +
            a2 * square(invariant[n] - inv0);
    }
}

model {
    sun_z ~ uniform(-100, 100);
    sun_vz ~ uniform(-100, 100);
    ln_hz ~ uniform(0, 8);
    Sigma ~ uniform(0, 1000);

    for (n in 1:n_stars) {
        if (is_inf(mu[n]) || is_nan(mu[n])) {
            target += negative_infinity();
            break;
        }
        target += normal_lpdf(elem[n] | mu[n], s);
    }
}
