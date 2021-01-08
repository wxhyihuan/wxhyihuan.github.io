---
layout: post
title: MetaTaxoOPU  R tool
---

```{r}

library("ape")
library("tidyverse")

#MNp4nj <- read.tree("p4_NJ_copy.tree")
MNp4nj <- read.tree("p4nj_sub_exract1.tree")

distval <- dist.nodes(MNp4nj)
distval[1:5, 1:5]

o <- plot(MNp4nj)
plot(MNp4nj, show.tip.label = F, x.lim = o$x.lim)
tiplabels(MNp4nj$tip.label[grep("Otu*", MNp4nj$tip.label)], grep("Otu*", MNp4nj$tip.label), adj = 0, col = "red", frame = 'none', bg = "none", font = 3, cex = 0.8)
tiplabels(MNp4nj$tip.label[grep("Otu*", MNp4nj$tip.label, invert = T)], grep("Otu*", MNp4nj$tip.label, invert = T), adj = 0, col = "black", frame = 'none', bg = "none", font = 1, cex = 0.8)

OPUnummark <- 1
OPUmark <- "OPU_"
NodeOPU <- paste(OPUmark, OPUnummark, sep = "")

MNp4nj$tip.label[MNp4nj$edge[,][which(MNp4nj$edge[, 1] == 112),]]


## 每个Otu序列所在的节点和节点路径
otu_tip_id <- grep("Otu*", MNp4nj$tip.label)
otutip_nodepath <- nodepath(MNp4nj)[otu_tip_id]
## 获取指定节点（191）下的所有tips label
test1 <- extract.clade(MNp4nj, 191)
## 获取Otu tip所在的上级节点
otutips_secondlevel_node <- lapply(otutip_nodepath, function(el) {
  el[length(el) - 1]
})



library("ape")
library("tidyverse")

#MNp4nj <- read.tree("p4_NJ_copy.tree")
MNp4nj <- read.tree("p4nj_sub_exract1.tree")

MNp4nj2 <- read.tree("p4nj_sub_exract3.ntree")
## 每个Otu序列所在的节点和节点路径
otu_tip_id <- grep("Otu*", MNp4nj$tip.label)
otutip_nodepath <- nodepath(MNp4nj2)[otu_tip_id]
## 获取指定节点（191）下的所有tips label
test1 <- extract.clade(MNpMNp4nj24nj, 191)
## 获取Otu tip所在的上级节点
otutips_secondlevel_node <- lapply(otutip_nodepath, function(el) {
  el[length(el) - 1]
})

o2 <- plot(MNp4nj2)
plot(MNp4nj2, show.tip.label = F, x.lim = o2$x.lim)
tiplabels(MNp4nj2$tip.label[grep("Otu*", MNp4nj$tip.label)], grep("Otu*", MNp4nj$tip.label), adj = 0, col = "red", frame = 'none', bg = "none", font = 3, cex = 0.8)
tiplabels(MNp4nj2$tip.label[grep("Otu*", MNp4nj$tip.label, invert = T)], grep("Otu*", MNp4nj$tip.label, invert = T), adj = 0, col = "black", frame = 'none', bg = "none", font = 1, cex = 0.8)
nodelabels(node = c(unlist(otutips_secondlevel_node)), pch = 21, bg = "black", cex = 1)
nodelabels(MNp4nj2$node.label, adj = c(0.5), col = "black", frame = 'none', bg = "none", font = 1, cex = 0.8)

taxo_name_decidor <- function(consensus_linegae, unculted = FALSE) {
  levels_mark <- c("bacteria", "phylum", "class", "order", "family", "genus", "species")
  #unclasified <- "Uncult"
  consensus_linegae_level <- length(consensus_linegae)

  suffix <- ifelse(consensus_linegae_level < length(levels_mark),
    ifelse(unculted == TRUE,
      str_c("uncult", levels_mark[consensus_linegae_level], consensus_linegae[consensus_linegae_level], sep = "_"),
      str_c(levels_mark[consensus_linegae_level], consensus_linegae[consensus_linegae_level], sep = "_")),
      consensus_linegae[consensus_linegae_level])
  return(suffix)
}

NodeOPU_namer <- function(prefix = "OPU", OPUnummark = " ", suffix = " ") {
  NodeOPU_nam <- str_c(prefix = "OPU", OPUnummark, suffix, sep = "_")
  NodeOPU_nam <- str_replace(NodeOPU_nam, " ", "_")
  return(NodeOPU_nam)
}

OPUnummark <- 0
Node_OPU_Names <- c()

for (ele in unlist(otutips_secondlevel_node)) {
  uncult_mark <- FALSE
  temp_node_tipslabel_lineage <- c() #用来临时记录某节点在的分类的 Lineage
  OPUnummark <- OPUnummark + 1
  OPUnummarkstr <- ifelse(OPUnummark <= 9, paste("00", OPUnummark, sep = ""), ifelse(OPUnummark <= 99, paste("0", OPUnummark, sep = ""), OPUnummark))

  tmptre <- extract.clade(MNp4nj2, ele, root.edge = 0, collapse.singles = TRUE, interactive = FALSE)
  #cat("Node no. has :\n", paste(tmptre$tip.label, sep = "\n"), "\n")
  for (tip in tmptre$tip.label) {
    tmptiptax <- NULL
    len <- NULL
    if (str_detect(tip, "Unclassified")) {
      tmptiptax <- str_split(tip, "___", simplify = TRUE)
      len <- length(tmptiptax)
      lineage <- str_split(tip, "___", simplify = TRUE)[len]
      #cat(lineage, "\n")
      uncult_mark <- TRUE
      temp_node_tipslabel_lineage <- c(temp_node_tipslabel_lineage, lineage)
    } else {
      tmptiptax <- str_split(tip, "__", simplify = TRUE)
      len <- length(tmptiptax)
      #last_levelname <- str_split(tmptiptax, " ", simplify = TRUE)[2]
      #cat(tmptiptax,"\n")
      if (len > 2) {
        if (str_split(tmptiptax, " ", simplify = TRUE)[3] != "Unclassified") {
          last_levelname <- str_split(tmptiptax, " ", simplify = TRUE)[2]
          last_levelname <- str_replace(last_levelname, "_", " ")
          lineage <- str_split(tmptiptax, " ", simplify = TRUE)[len]
          lineage <- str_c(lineage, last_levelname)
          #uncult_mark <- str_split(tmptiptax, " ", simplify = TRUE)[2]
          uncult_mark <- FALSE
          temp_node_tipslabel_lineage <- c(temp_node_tipslabel_lineage, lineage)
        } else {
          #last_levelname <- str_split(tmptiptax, " ", simplify = TRUE)[2]
          #last_levelname <- str_replace(last_levelname, "_", " ")
          lineage <- str_split(tmptiptax, " ", simplify = TRUE)[len]
          uncult_mark <- TRUE
          temp_node_tipslabel_lineage <- c(temp_node_tipslabel_lineage, lineage)
        }
      } else {
        last_levelname <- str_split(tmptiptax, " ", simplify = TRUE)[2]
        last_levelname <- str_replace(last_levelname, "_", " ")
        lineage <- str_split(tmptiptax, " ", simplify = TRUE)[len]
        lineage <- str_c(lineage, last_levelname)
        uncult_mark <- FALSE
        temp_node_tipslabel_lineage <- c(temp_node_tipslabel_lineage, lineage)
      }
    }
  }
  # cat(temp_node_tipslabel_lineage, "\n")
  temp_consensus_linegae <- c()
  for (ele in temp_node_tipslabel_lineage) {
    if (!str_detect(ele, "^OTU_")) {
      ## 排除 OTU 序列的 lineage 信息
      tmplineage <- str_split(ele, "_", simplify = TRUE)
      if (tmplineage[length(tmplineage)] == "uncultured") {
        tmplineage = tmplineage[1:(length(tmplineage) - 1)]
      }
      #cat(tmplineage, "\n")
      if (is.null(temp_consensus_linegae)) {
        temp_consensus_linegae = tmplineage
      } else {
        ## 用交集的形式获取 一致性 lineage 信息 
        temp_consensus_linegae <- intersect(temp_consensus_linegae, tmplineage)
      }
    }
    #tmplineage <- str_replace(tmplineage, " ", "")
    #cat(temp_consensus_linegae, "\n")
  }
  taxo_name <- taxo_name_decidor(temp_consensus_linegae, uncult_mark)
  NodeOPU_namerstr <- NodeOPU_namer(prefix = "OPU", OPUnummark = OPUnummarkstr, suffix = taxo_name)
  #cat(NodeOPU_namerstr, "\n")
  Node_OPU_Names <- c(Node_OPU_Names, NodeOPU_namerstr)
  temp_node_tipslabel_lineage <- c() #清除临时记录的 lineage 

}

o2 <- plot(MNp4nj2)
plot(MNp4nj2, show.tip.label = F, x.lim = o2$x.lim)
tiplabels(MNp4nj2$tip.label[grep("Otu*", MNp4nj$tip.label)], grep("Otu*", MNp4nj$tip.label), adj = 0, col = "red", frame = 'none', bg = "none", font = 3, cex = 0.7)
tiplabels(MNp4nj2$tip.label[grep("Otu*", MNp4nj$tip.label, invert = T)], grep("Otu*", MNp4nj$tip.label, invert = T), adj = 0, col = "black", frame = 'none', bg = "none", font = 1, cex = 0.7)
nodelabels(node = c(unlist(otutips_secondlevel_node)), pch = 21, bg = "#0000FF55", cex = 0.8)
nodelabels(text = Node_OPU_Names, node = c(unlist(otutips_secondlevel_node)), adj = c(0.5), col = "black", frame = 'none', bg = "none", font = 1, cex = 0.7)

```