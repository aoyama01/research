
require(fasttime)
require(lubridate)
require(signal)
library(pracma)
library(tseriesChaos)

#install.packages("lubridate")
DIR <- "../../../data/心拍変動まとめ_copy"
FN <- list.files(DIR,pattern="\\.csv",recursive=TRUE,full.names=TRUE)
N<-length(FN)

DIR2 <- "../../../data/睡眠段階まとめ_copy"
FN2 <- list.files(DIR2,pattern="\\.csv",recursive=TRUE,full.names=TRUE)
N2<-length(FN2)
a<-0


# for (file in FN) {
#   if (!file.exists(file)) {
#     cat(sprintf("File does not exist: %s\n", file))
#   }
# }


for(i in 1:39){
  
  res <- try ( TMP <- read.csv(FN[i],skip=5,header=TRUE), silent=T )
  TIME.RRI <- fastPOSIXct(TMP$time,tz="Japan")-9*60*60
  
  TIME.START <- min(TIME.RRI)
  
  TIME.END <- max(TIME.RRI)
  
  RRI <- TMP$RRI
  RRI.max <- 1666
  RRI.min <- 333
  RRI.diff <- 200
  
  dt <- 0.5
  
  for(j in 1:2){
    if(j%%2==1){
      res <- try ( TMP3 <- read.csv(FN2[j+a],header=T), silent=T )
      Time <- TMP3$Time
      stage<-TMP3$Score
    }
    if(j%%2==0){
      res <- try ( TMP2 <- read.csv(FN2[j+a],header=F), silent=T )
      
      sTime.START<-fastPOSIXct(TMP2$V2[6],tz="Japan")-9*60*60
      sTime.END<-fastPOSIXct(TMP2$V2[7],tz="Japan")-9*60*60
      sDate<-fastPOSIXct(TMP2$V2[3],tz="Japan")-9*60*60
      
      TIME_s <- fastPOSIXct(paste(TMP2$V2[3],TMP3$Time),tz="Japan")-9*60*60
      
      for(q in 1:(length(TIME_s)-1)){
        if(TIME_s[q]>TIME_s[q+1]){
          break
        }
      }
      if(q>0){
        arTIME_s <- fastPOSIXct(paste(TMP2$V2[3],TMP3$Time[1:q]),tz="Japan")-9*60*60
        brTIME_s <- fastPOSIXct(paste(TMP2$V2[3],TMP3$Time[q+1:(length(TIME_s)-q)]),tz="Japan")-9*60*60+60*60*24
        rTIME_s <-c(arTIME_s,brTIME_s)
      }else if(q==0){
        rTIME_s <- fastPOSIXct(paste(TMP2$V2[3],TMP3$Time),tz="Japan")-9*60*60
      }
      
      if(length(RRI[TIME.RRI>sTime.START&TIME.RRI<sTime.END])>0){
        sRRI<-RRI[TIME.RRI>sTime.START&TIME.RRI<sTime.END]
        sTime<-TIME.RRI[TIME.RRI>sTime.START&TIME.RRI<sTime.END]
      }else{
        sRRI<-NA
        sTime<-NA
      }
      if(length(sTime)>1){
        n.RRI <- length(sRRI)
        D1.RRI <- c()
        D2.RRI <- c()
        D1.RRI[1] <- 0
        D2.RRI[n.RRI] <- 0
        D1.RRI[2:n.RRI] <- abs(sRRI[2:n.RRI]-sRRI[1:(n.RRI-1)])
        D2.RRI[1:(n.RRI-1)] <- D1.RRI[2:n.RRI]
        
        Q.rec <- sum(sRRI)/n.RRI
        
        #t.5min.sub <- seq(min(sTime), max(sTime) - 300, by = 5 * 60)
        
        # 5分毎に10秒ずらしたタイムスタンプを生成
        #t.10sec.sub <- unlist(lapply(0:29, function(x) t.5min.sub + x * 10))
        
        #t.5min.sub <- seq(min(sTime),max(sTime),5*60)
        t.10sec.sub <- seq(min(sTime), max(sTime), 5*60)
        tdiff<- difftime(min(sTime), max(sTime), units = "hours")
        n.10sec.sub <- length(t.10sec.sub)-1
        
        ave.5min <- c()
        sd.5min <- c()
        TP <- c()
        HF <- c()
        RMSSD <- c()
        LF <- c()
        VLF <- c()
        HFnorm <- c()
        LFnorm <- c()
        LFHF <- c()
        
        PS <- c()
        OPA_0.025<- c()
        OPA_0.05<- c()
        OPA_0.075<- c()
        OPA_0.1<- c()
        st<-c()
        nb<-c()
        fwhm<-c()
        PSDscale9<-c()
        PSscale9<-c()
        tPSscale9<-c()
        centroid_frequency<-c()
        bandwidth<-c()
        peak_freq<-c()
        peak_p<-c()
        mf_max<-c()
        mf_mean<-c()
        mf_sd<-c()
        slope<-c()
        OPA_scale9<-c()
        sharpness<-c()
        mfhf<-c()
        N1<-c()
        N2<-c()
        N3<-c()
        N4<-c()
        REM<-c()
        WAKE<-c()
        cv_value<-c()
        mad_value<-c()
        time_diff<-c()
        timediff<-c()
        for(m in 1:n.10sec.sub){
          tmp1 <- sRRI[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff & sTime >= t.10sec.sub[m] & sTime < t.10sec.sub[m+1]]
          time<-sTime[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff & sTime >= t.10sec.sub[m] & sTime < t.10sec.sub[m+1]]
          if(sum(tmp1)>=240000){
            timediff[m]<-tdiff
            time_diff[m]<- difftime(t.10sec.sub[m], t.10sec.sub[1], units = "hours")
            
            ave.5min[m] <- mean(tmp1,na.rm=TRUE)
            sd.5min[m] <- sd(tmp1,na.rm=TRUE)
            tmp.diff <- diff(tmp1,na.rm=TRUE)
            # 念のため異常値の除去
            tmp.diff[tmp.diff>RRI.diff] <- NA
            RMSSD[m] <- sqrt(mean(tmp.diff^2,na.rm=TRUE))
            RRI.resamp <- approx(sTime[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff],sRRI[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff],xout=seq(min(sTime[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff]),max(sTime[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff]),dt),method = "linear")$y
            TIME.resamp <- seq(min(sTime[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff]),max(sTime[sRRI > RRI.min & sRRI< RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff]),dt)
            RRI.resamp.5min <- RRI.resamp[( TIME.resamp >= t.10sec.sub[m]) & ( TIME.resamp < t.10sec.sub[m+1])]
            
            TP[m] <- var(tmp1,na.rm=TRUE)
            # パワースペクトルの計???
            psd <- spectrum(RRI.resamp.5min,plot=FALSE)
            # 周波数の調整
            psd$freq <- psd$freq/dt
            psd$spec <- psd$spec*dt
            psd.sum <- sum(psd$spec)
            # グラフ作成
            HF[m] <- TP[m]*sum(psd$spec[psd$freq > 0.15 & psd$freq <= 0.4])/psd.sum
            #　LFパワー [ms^2]
            LF[m] <- TP[m]*sum(psd$spec[psd$freq > 0.04 & psd$freq <= 0.15])/psd.sum
            #　VLFパワー [ms^2]
            VLF[m] <- TP[m]*sum(psd$spec[psd$freq <= 0.04 & psd$freq  > 0.003])/psd.sum
            #　VLFパワー [ms^2]
            HFnorm[m]<-100*HF[m]/(TP[m]-VLF[m])
            LFnorm[m]<-100*LF[m]/(TP[m]-VLF[m])
            LFHF[m]<-LF[m]/HF[m]
            
            
            # HF成分 (0.15 Hz < f < 0.4 Hz) の範囲を抽出
            hf_indices <- which(psd$freq > 0.15 & psd$freq < 0.4)
            hf_psd <- psd$spec[hf_indices]
            hf_freq <- psd$freq[hf_indices]
            
            # HF成分に対するパワースペクトルの最大値、平均値、標準偏差を計算
            hf_max <- max(hf_psd)
            hf_mean <- mean(hf_psd)
            hf_sd <- sd(hf_psd)
            
            
            # HF成分 (0.15 Hz < f < 0.4 Hz) の範囲を抽出
            mf_indices <- which(psd$freq > 0.18 & psd$freq < 0.275)
            mf_psd <- psd$spec[mf_indices]
            mf_freq <- psd$freq[mf_indices]
            
            # LF成分に対するパワースペクトルの最大値、平均値、標準偏差を計算
            mf_max[m] <- max(mf_psd)
            mf_mean[m] <- mean(mf_psd)
            mf_sd[m] <- sd(mf_psd)
            
            #1. 周波数帯域内のエネルギー分布
            mfhf[m] <- sum(mf_psd)/sum(hf_psd)
            
            #2. スロープ特性
            #fit <- lm(log10(mf_psd) ~ mf_freq)  # 線形回帰（パワーを対数変換）
            #slope[m] <- coef(fit)[2]  # スロープ
            
            
            
            #3. 中心周波数と拡がり
            #3.1. 中心周波数
            #centroid_frequency[m] <- sum(mf_freq * mf_psd) / sum(mf_psd)
            
            #3.2. 拡がり
            #bandwidth[m] <- sqrt(sum(mf_psd * (mf_freq - centroid_frequency[m])^2) / sum(mf_psd))
            
            #4. ピーク関連の特徴
            peaks <- which.max(mf_psd)
            #4.1. ピーク周波数
            peak_freq[m] <- mf_freq[peaks]
            #4.2. ピーク値
            peak_p[m] <- mf_psd[peaks]
            #4.3. ピーク幅 
            half_max_value <- peak_p[m] / 2
            above_half_max <- which(mf_psd >= half_max_value)
            fwhm[m] <- mf_freq[tail(above_half_max, 1)] - mf_freq[head(above_half_max, 1)]
            #4.4. ピークの尖り度
            sharpness[m] <- peak_p[m] / fwhm[m]
            
            
            #5. 時系列変動に基づく指標
            #5.1平均絶対偏差 (Mean Absolute Deviation, MAD)
            mad_value[m] <- mad(mf_psd)
            #5.2変動係数 
            #cv_value[m] <- mf_sd[m] / mean(mf_psd)
            
            #6. 非線形特性
            #6.1. 相空間再構成
            #delay_time <- 1
            #embedding_dim <- 3
            #reconstructed_data <- delayEmbed(mf_psd, m = embedding_dim, d = delay_time)
            #6.2. Lyapunov指数
            #6.3. フラクタル次元（Hurst指数）
            #hurst_result <- hurst(mf_psd)
            
            #7. 位相特性
            #hilbert_transform <- hilbert(mf_psd)
            
            # 位相を抽出
            #phase <- Arg(hilbert_transform)  # 位相成分（複素数の偏角）
            
            
            
            
            
            Ti1<-stage[rTIME_s >= t.10sec.sub[m] & rTIME_s < t.10sec.sub[m]+300]
            nb[m]<-i
            ttt<-c()
            t1<-0
            t2<-0
            t3<-0
            t4<-0
            t5<-0
            t6<-0
            t7<-0
            aaa<-0
            if(length(Ti1)>=1){
              for(ii in 1:length(Ti1)){
                if(Ti1[ii]=="W"){
                  t1<-t1+1
                }else if(Ti1[ii]=="N1"){
                  t2<-t2+1
                }else if(Ti1[ii]=="N2"){
                  t3<-t3+1
                }else if(Ti1[ii]=="N3"){
                  t4<-t4+1
                }else if(Ti1[ii]=="N4"){
                  t5<-t5+1
                }else if(Ti1[ii]=="R"){
                  t6<-t6+1
                }else if(Ti1[ii]=="N"){
                  t7<-t7+1
                }
              }
              ttt<-c(t2,t3,t4,t5,t6,t1,t7)
              aaa<-which.max(ttt)
            }
            st[m]<-aaa
            tttt<-c()
            N1[m]<-t2
            N2[m]<-t3
            N3[m]<-t4
            N4[m]<-t5
            REM[m]<-t6
            WAKE[m]<-t1
            
            
          }else{
            ave.5min [m]<- NA
            sd.5min [m]<- NA
            TP[m]<- NA
            HF[m]<- NA
            RMSSD[m]<- NA
            LF[m]<- NA
            VLF[m]<- NA
            HFnorm[m]<- NA
            LFnorm[m]<- NA
            LFHF[m]<- NA
            timediff[m]<- NA
            mf_max[m] <- NA
            mf_mean[m] <- NA
            mf_sd[m] <- NA
            mfhf[m] <-  NA
            #slope[m] <-NA
            #centroid_frequency[m] <-NA
            #bandwidth[m]<-NA
            peak_p[m] <- NA
            peak_freq[m]<- NA
            fwhm[m] <- NA
            sharpness[m] <-NA
            #cv_value[m]  <-NA
            mad_value[m]  <-NA
            nb[m]<-NA
            st[m]<-NA
            N1[m]<-NA
            N2[m]<-NA
            N3[m]<-NA
            N4[m]<-NA
            REM[m]<-NA
            WAKE[m]<-NA
            time_diff[m]<-NA
          }
          #plot(sTime,sRRI,"l")
        }
        
        
        DAT <- na.omit(data.frame(N1,N2,N3,REM,WAKE))
        
        #DAT <- na.omit(data.frame(nb,st,ave.5min,sd.5min,RMSSD,TP,HF,LF,VLF,HFnorm,LFnorm,LFHF,mf_max,mf_mean,mf_sd,mfhf,peak_p,peak_freq,fwhm,sharpness,mad_value,time_diff))
        write.table(DAT,"全データ911指標5分睡眠段階.csv", sep = ",", append=TRUE, quote=FALSE, col.names=F,row.names=FALSE) 
        
        
      }
      
    }
    
  }
  
  a<-a+2
  
  
}
