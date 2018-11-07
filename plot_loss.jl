using PyPlot
function sigmoid(x) # avoid overflow in calculating sigmoid
    y=zeros(size(x))
    for i=1:length(y)
        if x[i]<0
            y[i]=exp(x[i])/(1+exp(x[i]))
        else
            y[i]=1/(1+exp(-x[i]))
        end
    end
    return y
end

function logsigmoid(x) # avoid overflow in calculating log(sigmoid)
    y=zeros(size(x))
    for i=1:length(y)
        if x[i]<0
            y[i]=x[i]-log(1+exp(x[i]))
        else
            y[i]=-log(1+exp(-x[i]))
        end
    end
    return y
end
NNpred(weights,x) = sigmoid(weights'*x)

SquareLoss(c,pred) = begin; vc=vec(c); vp=vec(pred); sum((vc .- vp).^2)/length(c); end

LikLoss(c,pred) = begin; vc=vec(c); vp=vec(pred); -sum( vc.*log.(vp) .+ (1 .- vc).*log.(1 .- vp)
)/length(c); end

function LikLoss(c,weights,x) # avoid overflow in calculating likloss
    logpred=logsigmoid(weights'*x)
    log1mpred=logsigmoid(-weights'*x)
    return -sum( c.*logpred .+ (1 .- c).*log1mpred )/length(c)
end

N = 200 # number of training points
D = 10 # dimension of input
x = randn(D,N) # training inputs
# make some training data:
w_true=randn(D); c=zeros(Bool,N)
for n=1:N
    if NNpred(w_true,x[:,n]) .> 0.5
        c[n]=1
    end
end
# plot a 1D slice of the loss functions :
vec0=10*randn(D) # a random point
vec1=10*randn(D) # a random direction
vec2=10*randn(D)
I=100
Loss1=zeros(I,I); Loss2=zeros(I,I)

for i=1:I
    lambda_i=i/I
    for j=1:I
        lambda_j=j/I
        Loss1[i,j] = SquareLoss(c,NNpred(vec0+lambda_i*vec1+lambda_j*vec2,x))
        Loss2[i,j] = LikLoss(c,vec0+lambda_i*vec1+lambda_j*vec2,x)
    end
end
plot(Loss1); title("Square Loss"); figure(); plot(Loss2); title("Lik Loss")
