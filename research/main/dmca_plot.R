#
# Higher-order detrending moving-averagecross-correlation analysis
# 0次DMCAのデモ (ラグなし)
# 【実際の解析ではラグの推定が重要です】
#
######################
# 事前にパッケージをインストールすること
# install.packages("longmemo")
# install.packages("signal")
######################
require(longmemo)
require(signal)
######################
# ファイルのパスを設定
file_path_1 <- '../../data/プロアシスト脳波・心拍_copy/2018年度（男性・自宅・避難所・車中泊）/脳波/DA_sheet/181A_001_Powerと睡眠ステージ.csv'
file_path_2 <- '../../data/プロアシスト脳波・心拍_origin/2018年度（男性・自宅・避難所・車中泊）/心拍/DA_sheet.csv'

# 1つ目のファイルを読み込む
data_1 <- read.csv(file_path_1, fileEncoding = "shift-jis")

# 2つ目のファイルを、最初の5行をスキップして読み込む
data_2 <- read.csv(file_path_2, fileEncoding = "shift-jis", skip = 5)

# 必要な行をフィルタリング（1290から41820まで、30行ごとに取得）(Pythonだと1290から41820って書いた)
filtered_data_2 <- data_2[seq(1290, 41820, by = 30), ]

# クロス相関に必要な列を抽出：1つ目のファイルからDelta_Ratio、2つ目のファイルからRRI
x1 <- data_1$Delta_Ratio
x2 <- filtered_data_2$RRI

# 平均を0、標準偏差を1に標準化
x1 <- (x1 - mean(x1, na.rm = TRUE)) / sd(x1, na.rm = TRUE)
x2 <- (x2 - mean(x2, na.rm = TRUE)) / sd(x2, na.rm = TRUE)

# 時系列の長さ
n <- length(x1)

############################
par(mfcol=c(2,2),las=1,cex.axis=1.2,cex.lab=1.5,mar=c(5,5,3,1),cex.main=1.6)
plot(1:n-1,x1,"l",col=3,xaxs="i",xlab="i",ylab='Delta ratio',main="Delta ratio")
# lines(1:n-1,eps.common,col=2)
length(x2)
plot(1:n-1,x2,"l",col=4,xaxs="i",xlab="i",ylab='RRI',main="RR-interval")
# lines(1:n-1,eps.common,col=2)
############################
# DMAで解析するスケールは奇数のみ
# 何点にするか
n.s <- 20
# スケールの決定
s <- unique(round(exp(seq(log(5),log(n/4),length.out=n.s))/2)*2+1)
#########################
F1 <- c()
F2 <- c()
F12_sq <- c()
# 【STEP1　】時系列の積分
y1 <- cumsum(x1)
y2 <- cumsum(x2)
# 0次DMAとDMCA
for(i in 1:n.s){
  # Detrending
  y1.detrend <- y1 - sgolayfilt(y1,p=0,n=s[i],m=0,ts=1)
  y2.detrend <- y2 - sgolayfilt(y2,p=0,n=s[i],m=0,ts=1)
  F1[i] <- sqrt(mean(y1.detrend^2))
  F2[i] <- sqrt(mean(y2.detrend^2))
  F12_sq[i] <- mean(y1.detrend*y2.detrend)
}
rho <- F12_sq/(F1*F2)
#####
# plot
###
# 相互相関
plot(log10(s),rho,col=2,ylim=c(-1,1),main="Cross-correlation")
abline(h=c(-1,0,1),lty=2,col=gray(0.5),lwd=2)
# スケーリング
log10F1 <- log10(F1)
log10F2 <- log10(F2)
log10F12 <- log10(abs(F12_sq))/2
y.min <- min(log10F1,log10F2,log10F12)
y.max <- max(log10F1,log10F2,log10F12)
#
plot(log10(s),log10(F1),col=3,pch=2,ylab="log10 (F(s))",ylim=c(y.min,y.max),main="Scaling")
points(log10(s),log10(F2),col=4,pch=3)
points(log10(s),log10(abs(F12_sq))/2,col=2,pch=1)




