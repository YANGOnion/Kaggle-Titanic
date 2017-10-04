
getTitle=function(s){
  reg=sapply(s,FUN = function(x) return(gregexpr("\\s.+\\.",x)))
  return(sapply(1:length(s),FUN=function(x) substr(s[x],reg[x][[1]]+1,reg[x][[1]]+attr(reg[x][[1]],"match.length")-1)))
}
data[1:3,getTitle(Name)]
