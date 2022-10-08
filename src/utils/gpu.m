N = 8192;
A = single(rand(N,N));
B = single(rand(N,N));
start = clock();
C = A * B;
elapsedTime = etime(clock(), start);
disp(elapsedTime);
gFlops = 2*N*N*N/(elapsedTime * 1e+9);
disp(gFlops);