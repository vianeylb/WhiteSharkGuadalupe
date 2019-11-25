data {
  
  // OBSERVATION PROCESS
  
  int<lower=1> Tlen; // length of the observations across all individuals
  vector[Tlen] ID; // ID for individual tracks
  vector[Tlen] steps; // steps for all sharks
  vector[Tlen] angles; // angles for all sharks

  // STATE PROCESS

  int<lower=1> N_gps; // number of states for steps/turns


  //hierarchical steps
  int<lower=1> NindivID; // individual IDs
  int<lower=1, upper=NindivID> indivID[Tlen]; // individual IDs
}

parameters {
  
  // STEPS
  positive_ordered[N_gps] mu_step_hier; // mean of gamma - ordered
  vector<lower=0>[N_gps] sigma_mu_step_hier; // SD of gamma
  positive_ordered[N_gps] mu_step[NindivID];
  positive_ordered[N_gps] sigma_step;  
  
  // ANGLES
  // unconstrained angle parameters
  vector[N_gps] xangle;
  vector[N_gps] yangle;

  simplex[N_gps] initdist; // initial distribution for fine scale movements
  
  simplex[N_gps] moveTPM[N_gps]; 
}


transformed parameters {
  
  // shapes and rates for the state-dependent distributions for steps
  vector[N_gps] shape[NindivID]; 
  vector[N_gps] rate[NindivID];
  
  // constrained parameters for turning angle distributions
  vector[N_gps] kappa = sqrt(xangle .* xangle + yangle .* yangle);  
  vector[N_gps] loc;

  
  for(n in 1:N_gps){
    loc[n] = atan2(yangle[n], xangle[n]);
  }
  
  for(n in 1:NindivID){
    shape[n] = mu_step[n] .* mu_step[n] ./ (sigma_step .* sigma_step);
    rate[n] = mu_step[n] ./ (sigma_step .* sigma_step);
  }
  
}


model {
  
  
  // for the forward algorithm
  vector[N_gps] logp;
  vector[N_gps] logptemp;
  
  // transition probability matrix
  matrix[N_gps,N_gps] log_tpm_tr_move;

  // transpose
  for(i in 1:N_gps)
    for(j in 1:N_gps)
      log_tpm_tr_move[j,i] = log(moveTPM[i,j]);

  
  // -----------------------------------------
  // PRIORS
  // -----------------------------------------
  
  // steps - in meters
  // for 5 min period
  mu_step_hier[1] ~ normal(50, 5);
  mu_step_hier[2] ~ normal(150, 10);
  mu_step_hier[3] ~ normal(250, 10);
  sigma_mu_step_hier ~ student_t(3, 0, 1);
  sigma_step ~ student_t(3, 0, 1);

  // angles
  xangle[1] ~ normal(-0.5, 1); // equiv to concentration when yangle = 0
  xangle[2] ~ normal(2, 2);
  xangle[3] ~ normal(0, 1);
  yangle ~ normal(0, 0.5);

  for(i in 1:NindivID){
    mu_step[i] ~ normal(mu_step_hier, sigma_mu_step_hier);
  }
  
  // FORWARD ALGORITHM

    // likelihood computation
    for (t in 1:Tlen) {
        // initialise forward variable if first obs of track
        if(t==1 || ID[t]!=ID[t-1]){
          logptemp = initdist;     
            for(n in 1:N_gps){
            if(steps[t]>=0) 
                logptemp[n] = logptemp[n] + gamma_lpdf(steps[t] | shape[indivID[t],n], rate[indivID[t],n]);
            if(angles[t] >= -pi())
                logptemp[n] = logptemp[n] + von_mises_lpdf(angles[t] | loc[n], kappa[n]);
          }
        } else {
          for (n in 1:N_gps) {
            logptemp[n] = log_sum_exp(to_vector(log_tpm_tr_move[n]) + logp);
            if(steps[t]>=0) 
                logptemp[n] = logptemp[n] + gamma_lpdf(steps[t] | shape[indivID[t],n], rate[indivID[t],n]);
            if(angles[t] >= -pi())
                logptemp[n] = logptemp[n] + von_mises_lpdf(angles[t] | loc[n], kappa[n]);
          }
        }
        logp = logptemp;
        
        // add log forward variable to target at the end of each track
        if(t==Tlen || ID[t+1]!=ID[t])
            target += log_sum_exp(logp);
    }

}







