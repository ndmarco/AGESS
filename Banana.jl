using StanBase
set_cmdstan_home!("/Users/ndm34/Projects/cmdstan")

using StanSample, DataFrames, Stan

model = "
data {
  int N;
  array[N] real y;
}

parameters {
  real theta1;
  real theta2;
}

model {
  for (i in 1:N)
   y[i] ~ normal(theta1+theta2^2,1);
  theta1 ~ normal(0,1);
  theta2 ~ normal(0.5,1);
}
";

sm = SampleModel("banana", model);

data = Dict("N" => 100, "y" => randn(100) .+ 1);

t1 = time()
rc = stan_sample(sm; num_cpp_chains=1, num_chains=1, num_warmups=50000, num_samples=50000, data);
time() - t1 

if success(rc)
  df = read_samples(sm, :dataframe);
  df |> display
end

