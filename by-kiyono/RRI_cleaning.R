# 【重要】まず，RRIにデータを入れる
#######################
# 正常値の設定
RRI.max <- 1250
RRI.min <- 350
RRI.diff <- 150
#######################
# 時系列の長さ
n.RRI <- length(RRI)
# 異常値の除外
D1.RRI <- c()
D2.RRI <- c()
D1.RRI[1] <- 0
D2.RRI[n.RRI] <- 0
D1.RRI[2:n.RRI] <- abs(RRI[2:n.RRI]-RRI[1:(n.RRI-1)])
D2.RRI[1:(n.RRI-1)] <- D1.RRI[2:n.RRI]
time.RRI.rev <- time.RRI[RRI > RRI.min & RRI < RRI.max
                         & D1.RRI < RRI.diff & D2.RRI < RRI.diff]
RRI.rev <- RRI[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff
               & D2.RRI < RRI.diff]
######################
# プロット
plot(time.RRI.rev,RRI.rev,type="l",col=2,xlab="time", ylab="RRI [ms]")