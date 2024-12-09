# https://chaos-kiyono.hatenablog.com/entry/2022/07/25/212843
#

# 時系列の長さ (時系列は奇数長になります)
N <- 2^12
# 奇数に変換
N <- round(N/2)*2+1
# 半分
M <- (N-1)/2
#########################
# モデルの自己共分散関数を与える
# 例：AR(2)過程の自己共分散関数
AR2.model <- function(n,a1,a2,sig2){
  Cov <- c()
  Cov[1] <- sig2*(1-a2)/(1-a1^2-a2-a1^2*a2-a2^2+a2^3)
  Cov[2] <- Cov[1]*a1/(1-a2)
  for(k in 3:(n+1)){
    Cov[k] <- a1*Cov[k-1] + a2*Cov[k-2]
  }
  return(Cov)
}
# パラメタの設定
a1 <- 1.6
a2 <- -0.9
sig2 <- 1
# 【注意】[-M,M]区間ではなく，[0,2M-1]にしている
acov.model <- c(AR2.model(M,a1,a2,sig2),rev(AR2.model(M,a1,a2,sig2)[-1]))
lag <- c(0:M,-(M:1))
#########################
# 自己共分散関数のフーリエ変換
fft.model <- fft(acov.model)
PSD.model <- Re(fft.model)
f <- c((0:M)/N,-(M:1)/(N))
#########################
# 白色ノイズの生成
WN <- rnorm(N)
#　ホワイトノイズのフーリエ変換
fft.WN <- fft(WN)
#########################
# サンプル時系列の生成
fft.sim <-  sqrt(PSD.model)*fft.WN
x.sim <- Re(fft(fft.sim,inverse=TRUE))/N
#########################
# 結果の描画
par(mfrow=c(2,3),cex.main=1.5,cex.axis=1.4,cex.lab=1.7,las=1,mar=c(5,5,2,2))
########################################################################
# PSDの推定
# spansなし
tmp <- spectrum(x.sim,plot="false")
freq <- tmp$freq
psd <- tmp$spec
#
plot(freq[freq > 0],psd[freq > 0],"l",log="xy",col=4,pch=16,xlim=c(f[2],0.5),xaxs="i",xlab="f",ylab="Periodogram",main=paste("without spans"),lwd=2,xaxt="n",yaxt="n")
lines(f[f > 0],PSD.model[f > 0],col=2,lwd=2,lty=2)
# 対数目盛を描く
axis(1,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(1,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
axis(2,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(2,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
########
legend("bottomleft", legend=c("Analytical Power Spectrum"), col=c(2), lty=c(2),lwd=c(2))
########################################################################
# PSDの推定
# spans=5
tmp <- spectrum(x.sim,spans=5,plot="false")
freq <- tmp$freq
psd <- tmp$spec
#
plot(freq[freq > 0],psd[freq > 0],"l",log="xy",col=4,pch=16,xlim=c(f[2],0.5),xaxs="i",xlab="f",ylab="Periodogram",main=paste("spans=5"),lwd=2,xaxt="n",yaxt="n")
lines(f[f > 0],PSD.model[f > 0],col=2,lwd=2,lty=2)
# 対数目盛を描く
axis(1,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(1,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
axis(2,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(2,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
########
legend("bottomleft", legend=c("Analytical Power Spectrum"), col=c(2), lty=c(2),lwd=c(2))
########################################################################
# PSDの推定
# spans=15
tmp <- spectrum(x.sim,spans=15,plot="false")
freq <- tmp$freq
psd <- tmp$spec
#
plot(freq[freq > 0],psd[freq > 0],"l",log="xy",col=4,pch=16,xlim=c(f[2],0.5),xaxs="i",xlab="f",ylab="Periodogram",main=paste("spans=15"),lwd=2,xaxt="n",yaxt="n")
lines(f[f > 0],PSD.model[f > 0],col=2,lwd=2,lty=2)
# 対数目盛を描く
axis(1,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(1,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
axis(2,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(2,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
########
legend("bottomleft", legend=c("Analytical Power Spectrum"), col=c(2), lty=c(2),lwd=c(2))
########################################################################
# PSDの推定
# spans=c(5,5)
tmp <- spectrum(x.sim,spans=c(5,5),plot="false")
freq <- tmp$freq
psd <- tmp$spec
#
plot(freq[freq > 0],psd[freq > 0],"l",log="xy",col=4,pch=16,xlim=c(f[2],0.5),xaxs="i",xlab="f",ylab="Periodogram",main=paste("spans=c(5,5)"),lwd=2,xaxt="n",yaxt="n")
lines(f[f > 0],PSD.model[f > 0],col=2,lwd=2,lty=2)
# 対数目盛を描く
axis(1,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(1,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
axis(2,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(2,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
########
legend("bottomleft", legend=c("Analytical Power Spectrum"), col=c(2), lty=c(2),lwd=c(2))
########################################################################
# PSDの推定
# spans=c(15,15)
tmp <- spectrum(x.sim,spans=c(15,15),plot="false")
freq <- tmp$freq
psd <- tmp$spec
#
plot(freq[freq > 0],psd[freq > 0],"l",log="xy",col=4,pch=16,xlim=c(f[2],0.5),xaxs="i",xlab="f",ylab="Periodogram",main=paste("spans=c(15,15)"),lwd=2,xaxt="n",yaxt="n")
lines(f[f > 0],PSD.model[f > 0],col=2,lwd=2,lty=2)
# 対数目盛を描く
axis(1,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(1,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
axis(2,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(2,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
########
legend("bottomleft", legend=c("Analytical Power Spectrum"), col=c(2), lty=c(2),lwd=c(2))
########################################################################
# PSDの推定
# spans=c(15,15,15,15,15,15)
tmp <- spectrum(x.sim,spans=c(15,15,15,15,15,15),plot="false")
freq <- tmp$freq
psd <- tmp$spec
#
plot(freq[freq > 0],psd[freq > 0],"l",log="xy",col=4,pch=16,xlim=c(f[2],0.5),xaxs="i",xlab="f",ylab="Periodogram",main=paste("spans=c(15,15,15,15,15,15)"),lwd=2,xaxt="n",yaxt="n")
lines(f[f > 0],PSD.model[f > 0],col=2,lwd=2,lty=2)
# 対数目盛を描く
axis(1,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(1,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
axis(2,at=10^(-5:5)%x%(1:9),label=FALSE,tck=-0.02)
tmp<-paste(paste(sep="","expression(10^",-5:5,")"),collapse=",")
v.label<-paste("axis(2,las=1,at=10^(-5:5),label=c(",tmp,"),tck=-0.03)",sep="")
eval(parse(text=v.label))
########
legend("bottomleft", legend=c("Analytical Power Spectrum"), col=c(2), lty=c(2),lwd=c(2))
##########################################################