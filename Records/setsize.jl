using Plots, Statistics

Ms = [1024, 1536, 2048, 2560, 3072, 4096]

records = [2325.28 2198.48 2119.34;
           2149.00 2065.15 2022.02;
           1871.93 1879.13 1966.56;
           2323.84 2115.10 2032.55;
           2265.95 2058.00 2135.11;
           2424.90 2366.94 2218.01]

averages = mean(records; dims=2)
variation = std(records; dims=2)

plot(Ms, averages, ribbon=variation,
     xlabel="initial working set size",
     ylabel="sec",
     leg=false)

savefig("Records/setsize.pdf")
