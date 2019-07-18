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

    // upper sequence
    real<lower=-1, upper=1> b0;
    real<lower=-1, upper=1> b1;
    real<lower=-7, upper=1> ln_s2;

    // OMG the baroqueness
    real<lower=0, upper=2> f_x0;
    real<lower=-10, upper=0> f_k;
    real<lower=0, upper=0.1> f_a;
    // real f_b;
}

transformed parameters {
    real s = exp(ln_s);
    real s2 = exp(ln_s2);
    real hz = exp(ln_hz);
    real Sigma = exp(ln_Sigma);
    vector[n_stars] Ez;
    vector[n_stars] invariant;
    vector[n_stars] mu;
    vector[n_stars] mu2;

    vector[n_stars] f_vals;

    for (n in 1:n_stars) {
        Ez[n] = 0.5 * square(v_z[n]) + sech2_potential(z[n], G, Sigma, hz);
    }
    invariant = log(Ez) - mean(log(Ez));

    for (n in 1:n_stars) {
        mu[n] = a0 +
            a1 * (invariant[n] - inv0) +
            a2 * square(invariant[n] - inv0);

        mu2[n] = b0 + b1 * invariant[n];

        f_vals[n] = f_a + (0.9 - f_a) / (1 + exp(f_k * (invariant[n] - f_x0)));
    }
}

model {
    real lp1;
    real lp2;

    sun_z ~ uniform(-100, 100);
    sun_vz ~ uniform(-100, 100);
    ln_hz ~ uniform(0, 8);
    Sigma ~ uniform(0, 1000);

    ln_s ~ uniform(-7, 1);
    ln_s2 ~ uniform(-7, 1);
    inv0 ~ uniform(-3, 3);

    for (n in 1:n_stars) {
        if (is_inf(mu[n]) || is_nan(mu[n])) {
            target += negative_infinity();
            break;
        }

        lp1 = normal_lpdf(elem[n] | mu[n], s);
        lp2 = normal_lpdf(elem[n] | mu2[n], s2);
        target += log_mix(f_vals[n], lp2, lp1);

    }
}
