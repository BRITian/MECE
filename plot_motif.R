#install.packages("ggseqlogo")

setwd('E:\\F≈Ã\\R_workspace')
library(ggplot2)
library(ggplot2)
library(ggseqlogo)


url = 'E:/ø«æ€Ã«√∏/gradcam/csv'
)
ourl = "E:/ø«æ€Ã«√∏/gradcam/csv/1754_vactor_mean-46.csv"

ourl_blast = "E:/ø«æ€Ã«√∏/gradcam/csv-xin/1754-blast-46.csv"

data = read.csv(ourl)
data <- subset(data, select = -X )
names(data) <- c('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
data = t(data)

data_blast = read.csv(ourl_blast)
data_blast <- subset(data_blast, select = -X)
names(data_blast) <- c('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
data_blast = t(data_blast)
help(legend)

csl <- make_col_scheme(chars = c('A', 'C', 'D', 'E', 'F', 'G', 'H','I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'), 
                       groups = c("Hydrophobic", "Polar", "Acidic", "Acidic", "Hydrophobic", "Polar", "Basic", "Hydrophobic", "Basic", "Hydrophobic", "Hydrophobic", "Neutral", "Hydrophobic", "Neutral", "Basic", "Polar", "Polar", "Hydrophobic", "Hydrophobic", "Polar"), 
                       cols = c("#ffa400", "#0000ff", "#008000", "#008000", "#ffa400", "#0000ff", "#ff0000", "#ffa400", "#ff0000", "#ffa400", "#ffa400", "#df47fa", "#ffa400", "#df47fa", "#ff0000", "#0000ff", "#0000ff", "#ffa400", "#ffa400", "#0000ff"))
p1 = ggseqlogo(data,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(0, 69.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
p2 = ggseqlogo(data_blast,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(0, 69.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
p3 = ggseqlogo(data,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(69.5,139.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
p4 = ggseqlogo(data_blast,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(69.5,139.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
p5 = ggseqlogo(data,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(139.5,209.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
p6 = ggseqlogo(data_blast,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(139.5,209.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
p7 = ggseqlogo(data,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(209.5,278.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
p8 = ggseqlogo(data_blast,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(209.5,278.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))
#p = ggseqlogo(data_blast,col_scheme=csl,method="custom",seq_type="aa")+ylab("Probability")+xlab("Amino acid position")+xlim(c(550.5,620.5))+theme(plot.title = element_text(hjust = 0.5,size = 50, face = "bold"),axis.text=element_text(size=50,face = "bold"),axis.title.x=element_text(size=50,face = "bold"),axis.title.y=element_text(size=55,face = "bold"),legend.text= element_text(size=50))

p1=p1+guides(fill=F)
p2=p2+guides(fill=F)
p3=p3+guides(fill=F)
p4=p4+guides(fill=F)
p5=p5+guides(fill=F)
p6=p6+guides(fill=F)
#p7=p7+guides(fill=F)
#p8=p8+guides(fill=F)
p = gridExtra::grid.arrange(p1,p3,p5,p7,ncol=1)
p_blast = gridExtra::grid.arrange(p2,p4,p6,p8,ncol=1)
#p_blast = p_blast+theme(axis.text= element_text(size=20))
#p_blast = gridExtra::grid.arrange(p2,p4,ncol=1)
#p = gridExtra::grid.arrange(p1,p2,p3,p4,p5,p6,p7,p8,ncol=2)
p
#surl = 'C:/Users/user/Desktop/s_duizhao.pdf'
#ggsave(plot = p_blast,surl,height = 30,width = 60,dpi = 600, limitsize = FALSE)
surl = 'C:/Users/user/Desktop/s.pdf'
ggsave(plot = p,surl,height = 30,width = 60,dpi = 600, limitsize = FALSE)
surl = 'C:/Users/user/Desktop/s_blast.pdf'
ggsave(plot = p_blast,surl,height = 30,width = 60,dpi = 600, limitsize = FALSE)

