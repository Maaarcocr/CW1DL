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

# The values of the three coefficients are selected at random
    vec0=10*randn(D)
    vec1=10*randn(D)
    vec2=10*randn(D)
    
    # The number of steps in the [0, 1] interval to take
    I=100
    
    # The values for the two loss functions on 1 dimension
    Loss11=zeros(I);
    Loss12=zeros(I)
    
    # The values for the two loss functions on 2 dimensions
    Loss21=zeros(I,I);
    Loss22=zeros(I,I)
    
    for i=1:I
        # Compute the 1-dimensional error value for both losses
        lambda_i=i/I
        Loss11[i] = SquareLoss(c,NNpred(vec0+lambda_i*vec1,x))
        Loss12[i] = LikLoss(c,vec0+lambda_i*vec1,x)
        for j=1:I
            # Compute the 2-dimensional error value for both losses
            lambda_j=j/I
            Loss21[i,j] = SquareLoss(c,NNpred(vec0+lambda_i*vec1+lambda_j*vec2,x))
            Loss22[i,j] = LikLoss(c,vec0+lambda_i*vec1+lambda_j*vec2,x)
        end
    end

figure();
plot(Loss11);
title("Square Loss (1D)");

figure();
plot(Loss12);
title("Lik Loss (1D)")

figure();
plot(Loss21);
title("Square Loss (2D)")

figure();
plot(Loss22);
title("Lik Loss (2D)")
