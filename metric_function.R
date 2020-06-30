#Function to compute the desired metrics (Bias, RMSE, SI, Cor) all at once 
#22/06/2020
#Aurelien Callens


metrics_wave <- function(obs, pred, digits, var){
  m_pred <- mean(pred) 
  m_obs <- mean(obs)
  
  # Mean bias 
  biais <- m_pred - m_obs
  
  #Some results depend on the variable used
  #If we compute metrics for the wave direction we must be careful and be sure that
  # all results are between 0 and 360° (hence the modulo operations)
  
  if(var == "Dir"){
    #RMSE
    Rmse <- sqrt(mean(((obs - pred + 360 + 180 ) %% 360 - 180)^2))
    
    pred_centered <- ((pred - m_pred + 360 + 180 ) %% 360 - 180)
    obs_centered <- ((obs - m_obs + 360 + 180 ) %% 360 - 180)
    
    #Scatter index
    Si <- sqrt(sum(((pred_centered - obs_centered + 360 + 180 ) %% 360 - 180)^2) / sum(obs^2)) 
    
    #Correlation
    Cor <- mean(pred_centered * obs_centered)/(sd(pred)*sd(obs))
    
  } else {
    
    #RMSE
    Rmse <- sqrt(mean((pred- obs)^2))
    
    #Scatter index
    Si <- sqrt(sum(((pred - m_pred) - (obs - m_obs))^2) / sum(obs^2)) 
    
    #Correlation
    Cor <- mean((pred - m_pred) * (obs - m_obs))/(sd(pred)*sd(obs))
  }
  
  
  res <- rbind(biais,
               Rmse,
               Si,
               Cor)
  
  row.names(res) <- c("Biais", "Rmse", "SI", "Cor")
  return(round(res, digits))
}

