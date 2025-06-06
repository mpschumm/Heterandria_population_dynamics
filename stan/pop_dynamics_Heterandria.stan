
functions {
    vector  Het_pop_dynamics(real t,
           vector y,
           array[] real theta) {
    vector[2] dydt;
    // adults
    dydt[1] = theta[1] * y[2] - theta[2] * y[1];
    // juveniles
    dydt[2] = (y[1]/2)*exp(theta[4]*(1-((y[1]/2)/theta[5]))) - y[2] * theta[3] - theta[1] * y[2];
    return dydt;
  }
}

data {
  int<lower=0> N;
  array[2,N] int pre_harvest;
  array[N] vector[2] post_harvest;
  vector[N] delta_t;
  vector[5] parameters_scales;
  vector[5] parameters_mu;
}

// The parameters accepted by the model. 
parameters {
  real log_s;
  real log_m_A;
  real log_m_J;
  real log_alpha;
  real log_beta;
}

transformed parameters {
  vector[5] theta;
  // repeat centering/scaling for following lines after where you've done it for log_s
  theta[1]= exp(parameters_scales[1]*log_s+parameters_mu[1]);
  theta[2]=exp(log_m_A);
  theta[3]=exp(log_m_J);
  theta[4]=exp(log_alpha);
  theta[5]=exp(log_beta);
  array[N] vector[2] y_hat;
  for (i in 1:N) {
    array[2] real times;
    times[1] = 0.0;
    times[2] = delta_t[i];
    y_hat[i] = ode_rk45(Het_pop_dynamics, post_harvest[i], 0.0, times, theta);
    // line from previous work - ' array[N_times] vector[2] mu = ode_rk45(lvpp, y0, t0, times, theta); '
  }
}

// The model to be estimated. We model the output
// 'post_harvest' to be Poisson distributed (ideally, going forward, Neg Binom distributed).
model {
  /// define priors
  target+=std_normal_lpdf(log_s);
  target+=std_normal_lpdf(log_m_A);
  target+=std_normal_lpdf(log_m_J);
  target+=std_normal_lpdf(log_alpha);
  target+=std_normal_lpdf(log_beta);
  for (i in 1:N) {
    for (stage in 1:2) {
      target+= poisson_lpmf(pre_harvest[stage, i] | y_hat[i,stage]);
    }
  }
}

