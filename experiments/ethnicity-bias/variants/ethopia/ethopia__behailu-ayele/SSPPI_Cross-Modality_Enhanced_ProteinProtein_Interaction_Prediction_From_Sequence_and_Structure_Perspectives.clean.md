# SSPPI Cross-Modality Enhanced ProteinProtein Interaction Prediction From Sequence and Structure Perspectives

Author: Behailu Ayele
University: Pennsylvania State University


## Abstract

Recent advances have shown great promise in mining multimodal protein knowledge for better protein-protein interaction (PPI) prediction by enriching the representation of proteins. However, existing solutions lack a comprehensive consideration of both local patterns and global dependencies in proteins, hindering the full exploitation of modal information. Additionally, the inherent disparities between modalities are often disregarded, which may lead to inferior modality complementarity e GLYPH&lt;11&gt; ects. To address these issues, we propose a cross-modality enhanced PPI prediction method from the perspectives of protein sequence and structure modalities, namely SSPPI. In this framework, our main contribution is that we integrate both sequence and structural modalities of proteins and employ an alignment and fusion method between modalities to further generate more comprehensive protein representations for PPI prediction. Specifically, we design two modal representation modules ( Convformer and Graphormer ) tailored for protein sequence and structure modalities, respectively, to enhance the quality of modal representation. Subsequently, we introduce a Cross-modality enhancer module to achieve alignment and fusion between modalities, thereby generating more informative modal joint representations. Finally, we devise a cross-protein fusion (CPF) module to model residue interaction processes between proteins, thereby enriching the joint representation of protein pairs. Extensive experimentation on four benchmark datasets demonstrates that our proposed model surpasses all current state-of-the-art (SOTA) methods. The source codes are publicly available at the following link https: // 
## NOMENCLATURE

| Notation | Definition |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| G | Input 2-D protein graph. |
| V ; E | Set of vertices / edges in the protein graph. |
| l | Number of residues in the initial protein sequence modality. |
| n | Number of residues in the initial protein structure modality. |
| m | Number of patches in the patch-level pro- tein sequence representation. |
| d | Dimension of the embedding. |
| t | Graph di GLYPH<11> usion order of the random walk positional encoding. |
| s 2 R 1280 , | Residue / sequence representation of the |
| S 2 R l × 1280 | protein sequence modality. |
| e 2 R d , | Final patch / sequence embedding of the |
| E 2 R m × d | protein sequence modality. |
| M | Convolution kernel of the TextCNN. |
| Q ; K ; V | Query / key / value vector in the multihead attention. |
| x 2 R 1280 , | Node / graph representation of the protein |
| h 2 R d , | Final node / graph embedding of the pro- |
| H 2 R n × d | tein structure modality. |
| W ; b | Weight / bias parameter vector of the linear transformation. |
| A 2 R n × n , | Original / high-order adjacency matrix of the protein graph. |
| A 0 2 R n × n | Degree matrix of the protein |
| D 2 R n × n 2 | graph. Attention vector in the |
| a 2 R d , q | graph attention / cross-protein fusion attention. |
| P 2 R m × d | Cross-modality joint representation of the protein. |
| p 2 R d GLYPH<11> G , GLYPH<11> L | Representation of the protein pair. Attention score in global attention / graph attention. |
| ! | Frequency parameter to generate sinu- soidal positional encoding. |
| GLYPH<13> | Decay factor to calculate the high-order adjacency matrix. |
| GLYPH<21> | Scale parameter to balance the loss. |
| y | Prediction value of the model (a scalar for binary classification task or regression task; a vector for multi-class classification |

See https: // www.ieee.org / publications / rights / index.html for more information.

## I. INTRODUCTION

A S ONE of the most important biomolecules within cells, proteins play a crucial role not only in the construction of various cellular components but also in participating in signal transduction, gene transcription, immune response, and other key biological processes through complex interaction networks [1], [2]. A comprehensive dictionary of protein-protein interactions (PPIs) could contribute to understanding the intricate biological pathways within organisms, providing theoretical foundations for disease treatment and drug development [3], [4], [5]. With the continuous advancement of high-throughput sequencing technologies, costly and timeconsuming wet experimental methods to determine unknown PPIs have become increasingly inadequate to address the massive growth in demand. Consequently, the computational approaches, leveraging their e GLYPH&lt;14&gt; ciency and inexpensiveness, are gradually becoming the mainstream method to solve PPIs automatically [6], [7], [8], [9].

Early in silico approaches for predicting PPIs are primarily focused on machine learning (ML) methods. After being trained on predefined PPI datasets, ML models can be utilized for e GLYPH&lt;11&gt; ectively identifying unknown PPIs, making them more adaptable to high-throughput interaction prediction tasks compared to traditional wet lab methods. To ensure the reliability of ML models, researchers in this stage strive to update and optimize new algorithms from various perspectives [10], [11], [12], [13]. For example, MCD-SVM proposed a support vector machine (SVM)-based PPI prediction method and designed a novel protein feature representation, which could also take note of the sequentially distant but spatially close residues [14]. RF-LPQ treated the physicochemical property matrix of proteins as an image and applied the local phase quantization (LPQ) method to further extract high-quality protein descriptors, which were then fed into a random forest classifier for predicting PPIs [12]. Although these well-designed ML-based methods have achieved acceptable results in PPI prediction, their feature engineering still requires expert experience or complex domain knowledge.

As computational power continues to evolve, deep learning (DL) methods capable of adaptively extracting features from data have been extensively applied to study PPIs in recent years. Based on the protein information sources utilized by the DL-based models, developments in this field can be primarily classified into three aspects: sequence-, structure-, and multimodal information-based methods.

## A. Sequence-Based Methods

Sequence is the primary structure of proteins and also the most easily accessible modality of protein currently [15], [16]. The sequence-based DL methods treat protein sequences as textual data and employ natural language processing (NLP) algorithms to mine potential association patterns among residues, thereby obtaining rich semantic information of proteins for PPI prediction [17], [18], [19], [20]. For instance, DPPI achieved good performance by simply feeding the sequence profile obtained from multiple sequence alignment (MSA) into the stacked convolutional neural network (CNN)

modules [19]. Gui et al. [20] abandoned the complex feature preprocessing step by randomly encoding the sequences, and used a combination of CNNs and LSTMs for feature extraction. Subsequently, PIPR proposed to replace the residue encoding with word embeddings collected from Word2Vec [21] and designed a Siamese residual recurrent CNN (RCNN) for capturing contextual semantic information in sequences [22]. These sequence-based DL methods both show a good predictive performance and provide a more convenient solution for high-throughput PPI prediction problems. However, proteins are complex biomolecules with diverse structures that dictate their functions, and these sequence-based methods struggle to model the structural variations between proteins.

## B. Structure-Based Methods

Due to the labor-intensive nature of determining protein spatial structures, structure modality data for proteins are relatively scarce compared to sequence modality data. In recent years, the emergence of the structure prediction model AlphaFold2 [23] has greatly bridged this gap, leading to the development of a plethora of structure-based DL methods for PPI prediction [24], [25]. PPI-GNN introduced a novel approach that transforms the spatial structure of proteins into a topological graph by computing residue distances and then utilized graph neural networks (GNNs) to extract key structure information relevant to PPIs [26]. This topological graph conversion partly preserved the geometric features of proteins while significantly reducing computational complexity. Building upon this method, Huang et al. [27] further employed protein structure data to analyze surface characteristics, secondary structure features, and other geometric attributes of proteins, which were integrated as node features in the topological graph to enhance the model's perception and understanding of protein structure information. In addition, there are several PPI research methods based on protein spatial structures, which typically use the complete 3-D structure of proteins as input and extract spatial geometric representations or conformational physicochemical features to enhance prediction accuracy in downstream PPI-related tasks. For example, MutaBind2 [28] calculated six types of physicochemical features (such as van der Waals interaction energy, solvation energy changes, etc.) from the 3-D structure of protein-protein complexes and performed well in the mutation-based protein-protein binding a GLYPH&lt;14&gt; nity prediction task ( GLYPH&lt;1&gt;GLYPH&lt;1&gt; G). On the other hand, GeoPPI [29] employed a self-supervised geometric learning framework to automatically extract atomic-level topological features (such as distances, dihedral angles, etc.) by reconstructing perturbed protein structures, achieving excellent performance in the multipoint mutation GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task. These structure-based methods have provided fresh insights into PPI prediction from the perspective of physical interactions. However, the interaction types between proteins are diverse and such physical interactions still do not encompass all possible types of protein interactions.

## C. Multimodal Information-Based Methods

In recent years, several PPI prediction methods based on multimodal protein data have emerged [30], [31], aiming to

leverage the complementarity of di GLYPH&lt;11&gt; erent data modalities to enrich protein feature representations and enhance the overall comprehensiveness and accuracy of prediction models. Jha and Saha [32] proposed a novel multimodal approach for PPI prediction, which involved diverse modal features from protein structure and sequences. This method employed voxelbased visualization of the protein's 3-D structure and utilized ResNet50 to obtain the volumetric representation of the protein, which was then merged with the autocovariance and conjoint triad sequence features to form a multimodal representation of the protein for PPI prediction. Similarly, TAGPPI also developed a sequenceand structure-based PPI models [33], in which the structural characteristics were derived from the protein topological graph using a graph attention network (GAT) and the sequence features were extracted from the protein sequence by TextCNN [34]. While these multimodal PPI prediction methods have achieved state-of-the-art (SOTA) performance compared to unimodal methods, further research is still needed in the following three aspects.

- 1) Unimodal representations fail to consider both local and global dependencies between residues. The interactions between residues in a protein sequence or structure are not only the result of coupling between adjacent residues but are also influenced by the overall protein structure [35], [36]. Therefore, if only local or global dependencies are considered, the unimodal representation may not be comprehensive.
- 2) Multimodal joint representation fails to achieve alignment between modalities. Protein modalities have di GLYPH&lt;11&gt; erent data composition structures and feature extraction methods. If cross-modal alignment is not achieved before modality representation fusion to eliminate modality di GLYPH&lt;11&gt; erences, the complementarity between modalities may be a GLYPH&lt;11&gt; ected.
- 3) Cross-protein joint representation fails to model residue interactions between protein pairs. The formation of PPIs usually involves interactions between specific residues on the surfaces of proteins [37], [38]. If protein pair representations are simply concatenated, important residues may be overlooked, leading to a compromised quality of the interaction representation.

In this article, we propose a novel multimodal (sequence and structure) DL framework for predicting PPIs, which contains four major modules to address the problems of existing multimodal PPI prediction methods: the Convformer module, Graphormer module, Cross-modality enhancer module, and cross-protein fusion (CPF) module. The overall framework is shown in Fig. 1. Specifically, the Convformer and Graphormer modules enhance the global awareness of the interactions between residues by incorporating self-attention mechanisms into the existing CNN and GNN networks and further enrich the feature representation of sequence and structure modalities. Then, a carefully designed Cross-modality enhancer module is used to align and fuse the sequence and structure modality representations, thereby enhancing the complementarity of the multimodal representations. Finally, a mutual attention mechanism is introduced in the CPF module to enhance the perception of residue interactions across proteins. Moreover, to reduce the di GLYPH&lt;11&gt; erences between modalities, the initial embeddings for the sequence and structure modalities are obtained from the same source, which is extracted from a pretrained protein language model (PLM), i.e., ESM-2 [39]. Meanwhile, we use a Siamese network architecture in the encoder to reduce the di GLYPH&lt;14&gt; culty of cross-protein interaction, under which two input proteins are fed into the same encoder and mapped to an identical semantic space. Evaluation results showed that the proposed method achieved SOTA performance on four benchmark datasets.

## II. METHODS

## A. Preliminaries

For better readability, we first introduce the following important notations and definitions adopted in this study before detailing our method. We represent a scalar with lowercase letters (e.g., d for the hidden dimension of network), a vector with lowercase bold letters (e.g., h 2 R d ), and a matrix with uppercase bold letters (e.g., X 2 R n × d ), and use italic letters to denote special symbols (e.g., str for the protein structure modality). The summary of the key notations used in this article can be found in Nomenclature.

- 1) Problem Statement: PPI prediction involves multiple distinct tasks: 1) the binary classification task of determining whether an interaction occurs between a given pair of proteins and 2) the multiclass classification task of identifying the specific type of interaction between the proteins. Additionally, we extend the scope of PPI prediction by incorporating a third task focused on predicting the change in protein binding a GLYPH&lt;14&gt; nity ( GLYPH&lt;1&gt;GLYPH&lt;1&gt; G) resulting from mutations, which is framed as a regression task.

To address these tasks, we propose an end-to-end DL framework based on protein sequence and structure modality. Specifically, for a given protein pair &lt; pro1 , pro2 &gt; , our model takes their sequence and structural representations as input, denoted as &lt; (str1, seq1) , (str2, seq2) &gt; , where str refers to the structural representation and seq refers to the sequence representation of the proteins. In the GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task, pro1 and pro2 represent the protein complexes before and after the mutation, respectively.

For the binary classification task, the model outputs a scalar prediction y 2 f 0 ; 1 g , indicating the presence or absence of the PPI. For the multiclass classification task, the model outputs a vector of class probabilities or scores y , representing the likelihood of the interaction belonging to each of the predefined categories in the class space. For the regression task ( GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task), the model outputs a continuous value y , which corresponds to the predicted change in protein binding a GLYPH&lt;14&gt; nity due to a specific mutation.

- 2) Input Representation: For the protein sequence modality, the input data can be represented as S 2 R l × 1280 , where s i 2 R 1280 signifies the initial residue representation extracted from the pretrained ESM-2 and l denotes the length of the protein sequence. In terms of the protein structure modality, in this study, we adopt a previous method [40] to transform the 3-D protein structure into a 2-D topological graph, i.e., given

Fig. 1. Schematic of the SSPPI architecture. The SSPPI mainly consists of four modules: Convformer , Graphormer , Cross-modality enhancer , and CPF . The Convformer and Graphormer modules are, respectively, employed for extracting the sequence and structural modal representations of the protein. Subsequently, these two modal representations are fed into the Cross-modality enhancer module to obtain the joint modal representation of the protein. Finally, the joint modal representations of the two proteins are collectively input into the CPF module to acquire the ultimate joint representation of the protein pair for predicting PPI.

<!-- image -->

a protein structure, it can be defined as G = ( V ; E ), where V is the set of vertices (amino acids), and E represents the set of edges (contact map). In this scenario, the input data for the structure modality comprise the feature matrix X 2 R n × 1280 and adjacency matrix A 2 R n × n of the protein graph, where x i 2 R 1280 is also the ESM-2 embedding.

3) Multihead Attention: The multihead attention (MHA) mechanism is utilized multiple times within the proposed model. To ensure coherence in subsequent formulaic descriptions, we provide a preliminary definition of its general form. The MHA mechanism endeavors to capture the global correlations among various positions within the sequence by computing their attention distributions, which would facilitate a more e GLYPH&lt;11&gt; ective representation of the sequence. The prevalent approach for attention computation is the scaled dot-product attention [41], as depicted in the following equation:

<!-- formula-not-decoded -->

where d is the feature dimension of the input sequence. Building on the scaled dot-product attention, the MHA mechanism introduces multiple heads in parallel to learn di GLYPH&lt;11&gt; erent attention weights, thereby enabling the model to capture richer information across various semantic spaces. The equation can be represented as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where W C is a linear transformation parameter. Depending on the input source, the MHA mechanism can be further

classified into two types: multihead self-attention (MHSA) and multihead cross-attention (MHCA), as illustrated by the following equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where A and B are two sequence representations from di GLYPH&lt;11&gt; erent sources.

## B. Sequence Modality Representation

The protein sequence contains rich sequence information, including motifs [42], conserved sites [43], and structure domain [44] information, which are closely related to protein function. In order to more fully exploit the e GLYPH&lt;11&gt; ective protein information in the sequence, we designed a sequence feature encoder combining CNN with MHSA, named Convformer .

1) Local Feature: Specifically, one zero-padding TextCNN layer was adopted to extract local features of protein sequences, aiming to endow the model with the capability to recognize short sequence pattern (e.g., motifs) that occurs consecutively within the sequence [42]. Given a representation of input sequence S 2 R l × 1280 , the local features S 0 2 R l × d can be extracted by the following equation:

<!-- formula-not-decoded -->

where M 2 R c × d is a weight-sharing kernel with d channels, c is the kernel size, and b is a bias vector.

2) Global Feature: Although convolutional operations can cover some crucial residues which are continuous in protein sequence, the residues at important sites may actually be dispersed [45]. Therefore, introducing self-attention mechanisms to enhance the model's perception of long-range dependencies is essential. However, the full connectivity characteristic of self-attention mechanisms may overshadow the locally extracted features by the CNN. Hence, we design a patchlevel self-attention to balance the relationship between local and global information. First, we partitioned the sequence representation in residue-level S 0 2 R l × d into a series of residue blocks (patches) and obtained the sequence representation in patch-level S 00 2 R m × d based on the following equation:

<!-- formula-not-decoded -->

where m is the number of patches and k is the length of a patch. Then, a sinusoidal positional encoding PE patch = [ pe 1 ; pe 2 ; pe 3 ; : : : ; pe m ] was solved and added to S 00 to obtain the position-aware sequence representation S 000 2 R m × d

<!-- formula-not-decoded -->

where ! k = (1 = 10 000 (2 k = d ) ) is a frequency parameter. Finally, a residual MHSA layer and a feedforward network (FFN) were employed to extract the global feature of patches and acquired the final sequence embedding E 2 R m × d

<!-- formula-not-decoded -->

where LN denotes a layer normalization layer. Through this approach, it is possible to achieve global interactions between patches while preserving local features within each patch, thereby obtaining a higher quality sequence embedding.

## C. Structure Modality Representation

In this study, we propose a novel GNN model, Graphormer , that is specifically designed to learn key structural information within a protein using the 2-D protein topological graph G = ( V ; E ) as input. The Graphormer we proposed primarily comprises two stages: global attention stage and graph attention stage. With this design, the global features learned by the global attention mechanism serve as the contextual information to guide the graph attention in e GLYPH&lt;11&gt; ectively capturing the structural information of the protein graph.

1) Global Attention Stage: During the global attention stage, to ensure that the model does not merely treat graph nodes as linear sequences for global feature extraction, we introduce two strategies: random walk positional encoding (RWPE) [46] and graph structure encoding, based on the selfattention mechanism, to enhance its perception of the overall graph structure. Specifically, given a protein graph G , we first performed random graph di GLYPH&lt;11&gt; usion based on its adjacency matrix A 2 R n × n to generate the RWPE matrix PE rw 2 R n × t

<!-- formula-not-decoded -->

where RW = AD GLYPH&lt;0&gt; 1 2 R n × n is the random walk operator, D 2 R n × n denotes the degree matrix of A , and t is the di GLYPH&lt;11&gt; usion steps. Next, we linearly fused the initial feature matrix X 2 R n × 1280 of the protein graph with the PE rw to obtain a structure-aware feature matrix X 0 2 R n × d

<!-- formula-not-decoded -->

where W X and W PE are learnable matrices and b X and b PE are bias vectors. Finally, the structure-aware feature matrix X 0 was then fed into the global attention mechanism for global feature extraction, as illustrated in the following equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where GLYPH&lt;11&gt; G i j represents the global attention score for node i and j , c is a learnable scalar, A 0 2 R n × n stands for the graph structure encoding calculated by (14), ALL represents the set of all nodes in the protein graph, h is the number of attention heads, and GLYPH&lt;27&gt; denotes a nonlinear activation function

<!-- formula-not-decoded -->

where A m represents the m -order adjacency matrix of protein graph G , GLYPH&lt;13&gt; m is a decay factor, P 1 m = 0 GLYPH&lt;13&gt; m = 1, and GLYPH&lt;13&gt; m 2 [0 ; 1].

2) Graph Attention Stage: After global attention, the new feature matrix X 00 2 R n × d of the protein graph encodes comprehensive contextual information. Then, a residual GAT was employed to further capture the structural information of the protein by means of edge attention aggregation. This layer can be defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where GLYPH&lt;11&gt; L i j represents the graph attention score for node i and j , a is an attention vector, W is a shared parameter matrix used to apply a linear transformation to the node feature, NBR denotes the set of all neighboring nodes of node i , and K is the number of attention heads. After these two stages, we obtained the final structural embedding H 2 R n × d of the protein graph.

## D. Cross-Modality Enhancer

Due to the modality di GLYPH&lt;11&gt; erences, directly fusing the sequence embedding E 2 R m × d and structure embedding H 2 R n × d may lead to information loss or mutual interference, thus a GLYPH&lt;11&gt; ecting the complementary e GLYPH&lt;11&gt; ects between modalities. Therefore, a Cross-modality enhancer module is designed to achieve feature alignment and fusion in this article, which consists of two parts: cross-modal alignment and cross-modal fusion.

1) Cross-Modal Alignment: To achieve feature alignment between protein sequence and structure modalities, we employ a contrastive learning framework designed to maximize the similarity between the global representations of the sequence and structure belonging to the same protein while minimizing the similarity between representations originating from di GLYPH&lt;11&gt; erent proteins. Specifically, we first performed global average pooling on the sequences embedding E 2 R m × d and structure embedding H 2 R n × d to obtain the global sequence representation e 2 R d and the global structure representation h 2 R d , respectively. Then, we utilized two modality-specific mapping blocks to map them to a new feature space. These processes are defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where W is the learnable parameter matrix and b is the bias vector. Finally, we introduced contrastive learning on the sequence-structure pairs to bring the feature spaces of the two modalities closer together.

The core alignment objective of our proposed contrastive learning is realized through the InfoNCE loss, which operates within each training mini-batch. Consider a mini-batch containing B distinct proteins. For any given protein i within this batch, its global sequence representation e i serves as an anchor. The corresponding positive sample is its own global structure representation h i . Crucially, the negative samples are the global structure representations h j ( j , i ) of all other proteins within the same mini-batch, resulting in B GLYPH&lt;0&gt; 1 negatives per anchor. Conversely, when the global structure representation h i is the anchor, the positive sample is its own global sequence representation e i , and the negatives are the global sequence representations e j ( j , i ) of all other proteins in the batch. Based on this formulation of anchors, positives, and negatives, we compute the InfoNCE loss symmetrically for both perspectives. The sequence-anchored loss for protein i encourages high similarity (measured by a function sim, such as cosine similarity) between the anchor e i and its positive h i while suppressing similarity to all negatives h j ( j , i )

<!-- formula-not-decoded -->

where the summation in the denominator P B j = 1 exp((sim( e i ; h j ) = T )) encompasses the structure representations h j of every protein j (indexed from 1 to B ) within the current mini-batch, explicitly including both the single positive sample ( j = i ) and all B GLYPH&lt;0&gt; 1 negative samples ( j , i ). The structure-anchored loss is defined analogously

<!-- formula-not-decoded -->

where the denominator P B j = 1 exp((sim( h i ; e j ) = T )) similarly sums over the sequence representations e j of every protein in the batch, including the positive and negatives. The hyperparameter T &gt; 0 is a temperature coe GLYPH&lt;14&gt; cient scaling the similarity scores. The total contrastive loss is the average of both sequence- and structure-anchored losses computed over all B proteins in the batch

<!-- formula-not-decoded -->

This symmetric loss function e GLYPH&lt;11&gt; ectively aligns the sequence and structure modalities by pulling representations of the same protein together in the shared latent space and pushing apart representations from di GLYPH&lt;11&gt; erent proteins.

2) Cross-Modal Fusion: After modal alignment, we introduce a multimodal fusion block (23) to further integrate modalities and obtain the final joint representation P 2 R m × d across modalities

<!-- formula-not-decoded -->

where FFN is a feedforward network, LN is a layer normalization layer, MHCA is an MHCA layer shown in (5), and MHSA is a multihead self-attention layer shown in (4).

## E. Cross-Protein Fusion

To capture the pairwise local interaction between the protein pair &lt; pro1 , pro2 &gt; , we apply a CPF module on their modal joint representation P (1) 2 R m × d and P (2) 2 R m × d . It mainly consists of two layers: one for modeling interactions of the protein pair, termed the interaction mapping layer, and another

for acquiring joint representations of the protein pair, termed the attention pooling layer. Specifically, we first calculated the multihead interaction map between two proteins as follows:

<!-- formula-not-decoded -->

where map 2 R m × m denotes the interaction scores, k denotes the number of head, W 2 R d × d is the parameter matrix, q 2 R d is an attention vector, and GLYPH&lt;14&gt; represents the Hadamard product. Note that there is a broadcast operation on the modal joint representation before the Hadamard product. Then, a columnwise and a row-wise averaging operations, shown in (25) and (26), were separately applied to map 2 R m × m , and thus, two attention score vectors s (1) 2 R m and s (2) 2 R m can be obtained, in which the element si denoted the attention sore of an entire protein to a single residue i in another protein

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where k is the number of attention heads. Finally, we can perform attention pooling on P (1) 2 R m × d and P (2) 2 R m × d based on their corresponding attention score vectors s (1) 2 R m and s (2) 2 R m to obtain the final protein joint representation p 2 R 2 d for the protein pair, which can be defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where jj denotes the concatenation operation and K is the number of attention heads we used.

## F. Prediction Module

1) PPI Prediction: The final protein joint representation p 2 R 2 d obtained from the CPF module is then fed into a classifier to predict the PPI, which can be denoted as follows:

<!-- formula-not-decoded -->

where y PPI denotes the prediction label and MLP is the classifier, which has three fully connected linear layers. The PPI prediction loss used in this study is the cross-entropy loss

<!-- formula-not-decoded -->

where n and m denote the number of samples and the number of PPI types, respectively, and y is the true PPI label. Finally, the optimization objective of our framework consists of two parts: the contrastive loss and the prediction loss

<!-- formula-not-decoded -->

where GLYPH&lt;21&gt; is an adjustable hyperparameter.

2) GLYPH&lt;1&gt;GLYPH&lt;1&gt; G Prediction: Considering that there is no interaction between the protein complex before and after mutation, in the GLYPH&lt;1&gt;GLYPH&lt;1&gt; G prediction task, we pool the protein modality joint representation obtained from the Cross-modality enhancer module and directly feed it into a GLYPH&lt;1&gt;GLYPH&lt;1&gt; G classifier for prediction, which can be denoted as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where MaxPooling denotes a max-pooling layer; p wild -type and p mutant represent the modality joint representations of the wild-type and mutant proteins, respectively; y GLYPH&lt;1&gt;GLYPH&lt;1&gt; G denotes the prediction label; and MLP is the classifier, which has three fully connected linear layers. The GLYPH&lt;1&gt;GLYPH&lt;1&gt; G prediction loss used in this study is the mean squared error loss

<!-- formula-not-decoded -->

where n denotes the number of samples and y s the true GLYPH&lt;1&gt;GLYPH&lt;1&gt; G label. Finally, the optimization objective in GLYPH&lt;1&gt;GLYPH&lt;1&gt; G prediction task is as follows:

<!-- formula-not-decoded -->

where GLYPH&lt;21&gt; is an adjustable hyperparameter.

## III. EXPERIMENTS

## A. Datasets

The PPI datasets selected to evaluate our model are the Yeast [47], Multi-species [22], and Multi-class [33] datasets, which are widely recognized and commonly used benchmark datasets. Among these three benchmark datasets, Yeast and Multi-species are binary classification datasets, while the Multi-class is a multiclass classification dataset. In addition to random splitting, we also performed Homology identity-based splitting and Cold-start splitting on the multi-species dataset to further assess the model's generalization capability, as detailed in the Dataset section of the supplementary materials.

Additionally, a more challenging task of predicting the impact of mutations on protein binding a GLYPH&lt;14&gt; nity is adopted to evaluate the performance of the models. This task aims to predict the change in the binding free energy ( GLYPH&lt;1&gt; G) of protein complexes before and after mutation, and is hence referred to as the GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task. In this study, we collected a large number of GLYPH&lt;1&gt;GLYPH&lt;1&gt; G samples from the SKEMPI database [48] to construct a GLYPH&lt;1&gt;GLYPH&lt;1&gt; G dataset, namely SKEMPI , which is a regression dataset. The statistical information of these datasets is presented in Table I, with more detailed descriptions provided in the supplementary materials.

## B. Evaluation Metrics and Experimental Settings

1) Evaluation Metrics: Seven commonly used metrics for the PPI prediction task, including accuracy, precision, sensitivity (i.e., recall), specificity, F 1-score, Matthews correlation coe GLYPH&lt;14&gt; cient (MCC), and area under the receiver operating characteristic curve (AUROC), were adopted to evaluate the

TABLE I STATISTICS OF THE BENCHMARK DATASETS USED IN THIS STUDY

proposed model's performance. For the GLYPH&lt;1&gt;GLYPH&lt;1&gt; G prediction task, we evaluated the model's performance using four metrics: Pearson correlation coe GLYPH&lt;14&gt; cient (PCC), Spearman rank correlation (Spearman), root-mean-squared error (RMSE), and mean absolute error (MAE). The detailed definitions of the above metrics can be found in the Section Evaluation metrics of the supplementary materials.

2) Training Detail and Data Splitting: The training and test sets for the three publicly available PPI datasets were obtained as in previous work [22], [33]. For the SKEMPI dataset we constructed, the training and test sets were randomly split in a 4:1 ratio. When conducting experiments, we train all methods on the training set and evaluate them on the test set, and all experimental results we reported were the means and standard deviations over five repetitions with the same configuration. Additionally, to determine the optimal hyperparameters, we conducted a fivefold cross validation on the training set of Yeast . Specifically, the training set of Yeast was further divided into five equal folds, with each fold being used as the validation set in rotation and the rest four folds as the training sets. The average of fivefold results was recorded as the final performance for a specific hyperparameter combination.

3) Computational Resources and Training Time: The training time and maximum GPU memory usage required to train a complete SSPPI model on four benchmark datasets are measured. Specifically, the training times for SSPPI on the Yeast , Multi-species , Multi-class , and SKEMPI datasets were 486, 2200, 4240, and 246 min, respectively, with corresponding maximum GPU memory usages of 19650, 11000, 20303, and 8298 MB. Additionally, we randomly selected 100 PPI samples from the Yeast dataset, each containing proteins with over 1200 residues, to evaluate the inference time of SSPPI on large samples. For these large samples, the total inference time of SSPPI was 0.97 s, with an average inference time of less than 10 ms per sample. All the results mentioned above are obtained using a computational platform equipped with a 13th Gen Intel 1 Core 2 i9-13900K CPU and an NVIDIA GeForce RTX 4090 GPU.

4) Hyperparameter Settings: The hyperparameters used in our proposed model are listed in Table II. Among these parameters, two critical parameters are determined by fivefold cross validation on the training set of the Yeast dataset, which are marked with '*' in the table. Note that the hyperparameters

1 Registered trademark.

2 Trademarked.

TABLE II HYPERPARAMETERS USED IN OUR PROPOSED MODEL

Fig. 2. Fivefold cross-validation results under di GLYPH&lt;11&gt; erent patch sizes in Convformer . The symbols, such as circles and triangles, represent the performance values of each fold.

<!-- image -->

of our proposed model were fine-tuned only on the Yeast dataset, and the determined optimal parameters were kept consistent across two additional datasets. As for the baseline models we retrained, their hyperparameters were set to the optimal parameters reported in their respective studies.

To balance the local and global features of the sequence, we proposed a patch-level self-attention mechanism in Convformer , which involves partitioning the sequence representations learned through CNN into a series of patches with a fixed block size [see (7)] and applying self-attention at the patch level. The size of the fixed block, namely patch size, will have an impact on the extraction of local and global features of the sequence, and thus, we treated it as a key hyperparameter for optimization. As shown in Fig. 2, when the patch size is 25 residues, the model achieves optimal performance in most metrics. This may be attributed to that the self-attention mechanism at a coarse-grained word level is incapable of capturing the global features of sequence when the patch size is too large. Conversely, when the patch size is too small, the self-attention mechanism may excessively emphasize the global features while overlooking the local features extracted by the CNN.

In this study, the topological graph is adopted to represent the protein structure modality, which is calculated based on the 3-D structure file of protein. Specifically, if the Euclidean coordinate distance between residues is less than a certain distance threshold, we consider these two residues to be adjacent; otherwise, they are not adjacent. This distance threshold determines the adjacency matrix of the protein topological graph, and thus, we treat it as another key hyperparameter

TABLE III PERFORMANCE COMPARISON OF SSPPI AND OTHER BASELINES ON THE YEAST DATASET

Fig. 3. Fivefold cross validation results under di GLYPH&lt;11&gt; erent distance thresholds for construction of protein graph. The symbols, such as circles and triangles, represent the performance values of each fold.

<!-- image -->

to fine-tune. The results, illustrated in Fig. 3, indicate that the model achieves optimal performance across various metrics when this distance threshold is set to 8 ˚ A.

## C. Experimental Results

1) Comparison on the Yeast Dataset: In order to demonstrate the superiority of our proposed model, we conducted performance comparison analysis of 12 baseline models on the Yeast dataset. It is worth noting that due to the unavailability of training code, some methods were not retrained in this study, and their experimental results were collected from [22]. As shown in Table III, our method outperforms other SOTA baselines in all seven metrics, including accuracy (98.93%), precision (98.98%), recall (98.88%), specificity (98.98%), F 1-score (98.93%), AUC (99.70%), and MCC (97.86%). Furthermore, according to the experimental results, we also observe that two multimodality-based methods (i.e., our proposed model and TAGPPI) achieved promising results on this dataset, demonstrating the beneficial impact of multimodal information in enhancing PPI prediction performance.

2) Comparison on the Multispecies Dataset Under Homology Identity-Based Condition: On the Multi-species dataset, we only selected seven baseline models that could be retrained for performance comparison, and these models also demonstrated significantly superior performance on the Yeast dataset. The experimental results are presented in Table VI. From the table, it can be observed that as the protein homology identity increases, the overall performance of the models also gradually improves. Regardless of any homology identity partition, our model outperforms other models in most metrics. Even on datasets with low homology identity, our model continues to perform exceptionally well. Particularly, when the homology identity is less than 1% and 10%, our model can still achieve performance exceeding 90% in terms of ACC ( 99.19% for 1% and 99.16% for 10%) and F 1-score ( 99.29% for 1% and 99.25% for 10%) metrics. All of these demonstrate the strong generalization capability of our model. Moreover, the standard deviations of the SSPPI model are lower than other compared methods in most cases, which proves the stability of our model.

3) Comparison on the Multi-Species Dataset Under Cold Start Condition: In this study, we also evaluated SSPPI and all the baseline models under on Multi-species under cold start conditions, and the experimental results are reported in Table IV. As expected, compared to random partitioning, the performance of all baseline models significantly declined under the S1 and S2 conditions, with ACC falling from 99% to less than 83% and 74%, respectively. However, our proposed method can still achieve impressive performance

TABLE IV PERFORMANCE COMPARISON OF DIFFERENT METHODS ON MULTI-SPECIES DATASETS

and pronounced advantages over suboptimal methods ( 87.97% versus 84.42% for S1 and 81.27% versus 74.62% for S2), demonstrating its strong robustness and great practical value in real-world applications.

4) Comparison on the Multi-Class Dataset: We further evaluate the ability of our proposed model on a PPI type prediction task, i.e., Multi-class dataset. In this dataset, true labels of samples correspond to the types of interactions

TABLE V PERFORMANCE COMPARISON OF DIFFERENT METHODS

## ON MULTI-CLASS DATASET

TABLE VI

## PERFORMANCE COMPARISON OF DIFFERENT METHODS ON THE SKEMPI DATASET

Fig. 4. Scatter plot of predicted GLYPH&lt;1&gt;GLYPH&lt;1&gt; G of di GLYPH&lt;11&gt; erent models versus experimental GLYPH&lt;1&gt;GLYPH&lt;1&gt; G values on the SKEMPI dataset.

<!-- image -->

between proteins, making this task more challenging compared to binary classification tasks. To demonstrate the robustness of our proposed model, we retrained our model along with seven baseline methods that we used on Multi-species dataset to conduct a comparison analysis and the experimental results are presented in Table V. From the table, it can be observed that our proposed model generally outperforms other cuttingedge methods in most cases.

- 5) Comparison on the SKEMPI Dataset: Finally, we evaluated the performance of our model and the baseline models on the more challenging SKEMPI dataset, and the experimental results are shown in Table VI and Fig. 4. From the results, it is evident that our model outperforms the other baselines across four metrics: PCC, Spearman, RMSE, and MAE. Moreover,

the model's predictions exhibit strong linear correlation with the true labels. These findings demonstrate that our model remains e GLYPH&lt;11&gt; ective on the more challenging GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task.

## D. Ablation Study

To investigate the individual role of each component in our proposed model, we conducted a series of ablation experiments by proposing and testing several variants of our model as follows.

- 1) SSPPI Without the w / o CME Module: In this ablation variant, we remove the Cross-modality enhancer module from the SSPPI model. The sequence modality representation E 2 R m × d and structure modality representation H 2 R n × d obtained from Convformer and Graphormer are fused to generate the final joint representation P 2 R m × d across modalities with a simply feature combination operation as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 2) SSPPI Without the w / o CPF Module: For the protein pair &lt; pro1 , pro2 &gt; , their cross-modal joint representations, i.e., P (1) 2 R m × d and P (2) 2 R m × d , are directly concatenated to form the joint representation of this protein pair.
- 3) SSPPI Without the Convformer (w / o CNF) Module: This variant utilizes a simple CNN encoder to extract the representation of the sequence modality, replacing the original Convformer module.
- 4) SSPPI Without the Graphormer (w / o GHF) Module: Building upon the original Graphormer module, we remove its global attention component and replace it with a GAT layer. In other words, this variant leverages only two GAT layers to extract the structure modality representation of proteins.
- 5) SSPPI Without the Sequence (w / o SEQ) Modality Branch: The entire branch for sequence modality representation is removed and this variant only utilizes structure modality information to predict PPIs.
- 6) SSPPI Without the Structure Modality Branch (w / o STR): The entire branch for structure modality representation is removed and this variant only utilizes sequence modality information to predict PPIs.
- 7) SSPPI Without the Siamese Architecture (w / o SIA): In this variant, the two proteins are fed into separate encoders, meaning that the parameters are no longer shared between the protein encoders.
- 8) SSPPI With PCPs: The original ESM-2 embedding is replaced by a 33-D representation of the residue physicochemical properties (PCPs), as shown in Table S5 of the supplementary materials.
- 9) SSPPI With SCE: The original ESM-2 embedding is replaced by the pretrained Seqvec embedding (SCE) used by TAGPPI [33].

The performance comparison of SSPPI with its nine ablated variants on the Yeast dataset is shown in Table VII. The experimental results indicate that our proposed model achieved

TABLE VII RESULTS OF ABLATION EXPERIMENTS ON THE YEAST DATASET

the best performance compared to these variants, and removing any module would degrade the model performance in terms of all six metrics. Therefore, it can be concluded that each module included in our model is e GLYPH&lt;11&gt; ective for the task of PPI prediction. Among seven module variants, the w / o SIA , cross-modality enhancer w / o (CME) variant, and w / o GHF exhibit more noticeable performance degradation compared to the original model, demonstrating the significance of the Siamese architecture, the Cross-modality enhancer module, and the Graphormer module. The w / o STR and w / o SEQ perform relatively worse than the other variants, indicating that the multimodal information of proteins is indeed superior to using only a single modality for PPI prediction.

In addition to exploring module variants, we also investigated the impact of di GLYPH&lt;11&gt; erent feature sources on model performance (i.e., with PCP and with SCE in Table VII. From the results, we can see that replacing the original ESM-2 embeddings with the pretrained, i.e., with SCE , results in a slight decline in model performance. However, when replaced with the basic residue PCPs (i.e., with PCP ), the model performance significantly decreases, highlighting the e GLYPH&lt;11&gt; ectiveness of the prior knowledge encoded in pretrained embeddings for PPI prediction. Furthermore, we conducted an ablation experiment on SSPPI using basic PCPs as input features, with the experimental results provided in Table S6 of the supplementary materials. Under the condition of using PCP, the w / o CME , w / o SEQ , and w / o STR variants showed the most significant decline, indicating that these three modules contribute the most to the model. Notably, the w / o SEQ variant outperformed w / o CME when using PCP features, suggesting that, in the absence of the Cross-modality enhancer module, using the structural modality alone can be even more e GLYPH&lt;11&gt; ective than using both modalities together, where the modalities act more like noise to each other. This further illustrates that our proposed Cross-modality enhancer module can e GLYPH&lt;11&gt; ectively align the modalities and improve their complementarity.

## E. Visualization Study

In this study, we designed a visualization experiment on the binary classification datasets (i.e., Yeast and Multi-species )

Fig. 5. Visualization results of joint representation of the protein pair in the test set of Yeast and Multi-species . The red points denote the representation of positive protein pairs when reduced to two dimensions, while the blue points represent the negative samples.

<!-- image -->

to explore the representation power of our proposed model. Specifically, we preserved the trained model from the training phase and extracted the high-dimensional joint representation of protein pairs learned on the test set for dimensionality reduction and visualization using t-SNE technology [51]. The visualization results are shown in Fig. 5, where red points represent positive protein pairs and blue points represent negative protein pairs. From the experimental results, it can be observed that the high-dimensional joint representations learned on Yeast and Multispeices both exhibit clear clustering after dimensionality reduction, and positive protein pairs are well separated from negative protein pairs. This indicates that the joint representation of protein pairs learned by our model involves distinct spatial distribution characteristics, which is highly beneficial for PPI prediction.

## F. Case Study

To demonstrate the generalization capability of our model, we collected some unseen data that never appeared in the dataset we used. Subsequently, we designed two experiments to assess the model's performance, i.e., binding surface coverage analysis and signaling pathway prediction.

1) Binding Surface Coverage Analysis: This experiment analysis is based on an empirical assumption that the interaction binding sites of proteins typically occur on their structural surfaces [52]. To perform an experimental analysis of binding surface coverage, several dimeric complexes were collected from the protein data bank (PDB) [53], and these dimeric complexes both manifested a physical complementarity in their structure, resembling the lock-and-key model. Then, the two protein chains of these dimeric complexes were treated as the protein pair &lt; pro1 , pro2 &gt; and fed into the proposed model trained on the Yeast or Multi-species dataset, yielding the interaction attention scores between them, denoted as s (1) 2 R m and s (2) 2 R m in (25) and (26). It is worth noting that our mutual attention is at the patch level, where m represents the number of patches in the protein. Thus, we select the top-2 residue patches with the highest attention scores and map them back to the original complex structure

Fig. 6. Visualization results of mutual attention on protein structures. The ID annotated below each subgraph corresponds to the PDB ID of the complex. In the figures, the yellow and blue regions depict the two protein chains of the dimeric complex, while the highlighted red areas represent the key protein structures that the mutual attention focuses on. Furthermore, to better illustrate the prediction results, two visualization styles (left: Cartoon and right: Surface ) in Mol* [54] are employed for each complex structure.

<!-- image -->

for visualization. The final visualization results are shown in Figs. 6 and S7 of the supplementary materials, where yellow and blue respectively represent the two protein chains of the dimer, and the residue patches ranked in the top-2 based on attention scores are highlighted in red. It can be observed that the important residue positions our model focuses on largely cover the structural surface of the complex. In addition, for the two interacting proteins in the complex, we performed a correlation analysis between the attention scores of all residue patches in one protein and the distances of these residue patches to the interacting protein. The detailed method can be found in the Section Correlation analysis of the supplementary materials. As shown in Fig. S9 of the supplementary materials, there is a negative correlation between the distribution of attention scores and the distribution of distance values (PCC = GLYPH&lt;0&gt; 0 : 37), indicating that SSPPI places greater attention scores on patches with smaller distance values. In other words, our model focuses more on the binding surface regions of the complex, thereby capturing potential binding site information.

2) Signaling Pathway Prediction: The diversity of PPI in signaling pathways, which include direct physical interactions and indirect functional associations, aligns better with the complexity of PPI networks in realistic biological scenarios. In order to more comprehensively assess the generalization ability of the model, we collected two Wnt-related pathway networks [55] (i.e., one-core network and crossover network) and utilized them as external test sets to evaluate the performance of our model. The one-core network is composed of interactions between a core gene CD9 and 16 other genes, and this signaling pathway is closely associated with cell viability and tumor suppression. For the crossover network, it is comprised of four main core genes and a total of 94 pairs of interactions among 75 genes. This signaling pathway plays a crucial role in tumor growth and tumor formation.

Fig. 7. Predicted results of the crossover network for the Wnt-related pathway. The circles colored in red indicate the core genes. The black lines are the true prediction and the red lines are the false prediction.

<!-- image -->

After data processing, the interaction data from these two signaling pathways were fed into our pretrained model on the Multi-species dataset for prediction. The prediction results for the one-core and crossover networks are shown in Fig. S10 of the supplementary materials and Fig. 7, where black lines represent correctly predicted interactions, while red lines indicate cases where the interactions actually exist but were predicted incorrectly. Experimental results demonstrate that our model achieves prediction accuracies of 100% (onecore network) and 97.9% (crossover network) for these two signaling pathways. This confirms the excellent performance and strong generalization ability of our model in real biological scenarios.

## IV. DISCUSSION

## A. Modal Data Acquisition

Despite the significant potential of multimodal fusion methods for PPI prediction, the increase in the number of modalities introduces additional data acquisition burdens in practical applications. In large-scale high-throughput PPI prediction scenarios, where massive amounts of protein samples need to be analyzed, it is necessary to gather various required modal information, which substantially increases the complexity of data preprocessing and computational overhead. As the number of modalities increases, the burden of data processing gradually intensifies, especially with the tougher requirements for standardization and quality control of each modality. Furthermore, the increase in the number of modalities also exacerbates the risk of missing modalities, particularly in cases where certain complex modal data are di GLYPH&lt;14&gt; cult to obtain. For instance, when screening potential PPIs for newly designed proteins, some modalities (e.g., protein structure,

dynamic interaction data, etc.) are often challenging to acquire, which limits the applicability of multimodal PPI models. Therefore, while multimodal PPI prediction models exhibit significant theoretical advantages, their practical application faces more complex challenges, including the increased burden of data acquisition and preprocessing, as well as the limitations imposed by missing modalities. To address these issues, future research is warranted on e GLYPH&lt;11&gt; ectively integrating multimodal data, improving data acquisition e GLYPH&lt;14&gt; ciency, and reducing the risk of modality missing.

## B. Model Generalization

In this study, we evaluate the model's generalization ability from the following four perspectives.

- 1) The cross-dataset generalization capabilities in PPI prediction. Our model achieved SOTA performance across three distinct PPI datasets: the small-scale singlespecies dataset ( Yeast ), the large-scale multispecies dataset ( Multi-species ), and the multiclass PPI interaction dataset ( Multi-class ), demonstrating its robust adaptability to diverse PPI data distributions.
- 2) The generalization under rigorous dataset partitioning strategies. Homology identity-based splitting and cold-start splitting are two representative challenging scenarios for the PPI prediction task. While SSPPI showed a slight decrease in these conditions, it still achieved acceptable performance and outperformed all the cutting-edge baseline methods, confirming the excellent robustness of the model under conditions of low homologous knowledge or cold-start scenarios.
- 3) The generalization on newly discovered PPI data. In addition to curated datasets, our model also outperformed other multimodal-based PPI methods (e.g., TAGPPI) on two external test sets consisting of only newly discovered PPI samples (i.e., Yeast-HighConf and Yeast-LowConf , see Section Generalization evaluation on the PPI task in the supplementary materials), proving its practical value in real-world application scenarios.
- 4) The cross-task generalization. In addition to the PPI task, the comprehensive evaluations (see Section Generalization evaluation on the GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task in the supplementary materials) on the GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task show that our model not only accurately predicts the impact of mutations on changes in protein-protein binding a GLYPH&lt;14&gt; nity in static conformations but also exhibits strong prediction robustness on GLYPH&lt;1&gt;GLYPH&lt;1&gt; G samples that include dynamic conformations. This confirms that our model maintains excellent generalization performance across tasks. All of these findings demonstrate that SSPPI has good generalization capability and potential for real-world applications.

## C. Limitation of Using Static Structures

The interaction between two proteins typically involves conformational changes to achieve dynamic shape matching and energy complementarity. However, most existing PPI or GLYPH&lt;1&gt;GLYPH&lt;1&gt; G datasets only provide static snapshots of protein (which describe a rigid docking scenario), and the structure-based predictive models trained on these datasets are not able to capture the dynamic patterns, particularly for the GLYPH&lt;1&gt;GLYPH&lt;1&gt; G task where protein flexibility and induced fit e GLYPH&lt;11&gt; ects play a crucial role. Due to the absence of dynamic conformations, we were unable to incorporate the dynamic information upon model training. As an initial attempt, a mini-test was performed on 50 dynamic conformations generated from ten PPIs via simulation, and the results showed that SSPPI generalized well on these dynamic conformations (see Section Evaluation on the dynamic conformation test set in the supplementary materials). In this regard, future work should consider integrating ensemble-based structural representations or performing molecular dynamic simulations to better account for protein flexibility in PPI predictions.

## V. CONCLUSION

In this study, we propose a cross-modality enhanced protein representation framework SSPPI for PPI prediction. Within the entire SSPPI framework, we introduce four modules: the Convformer module for representing sequence modality, the Graphormer module for representing structure modality, the Cross-modality enhancer module for modal fusion, and the Cross-protein fusion module for protein interaction. First, in terms of modal representation, both the Convformer and Graphormer modules enhance the model's perception of global protein information by introducing global self-attention on the basis of CNN or GNN, thereby extracting higher quality sequence and structure modality representations. Next, the Cross-modality enhancer module we propose achieves e GLYPH&lt;11&gt; ective alignment and fusion across modalities through contrastive learning and cross-attention, generating more informative joint protein modality representations. Finally, in the protein interaction stage, our Cross-protein fusion module further enhances information exchange between proteins to enrich the final joint protein representations. Extensive experiments on four benchmark datasets demonstrate that our proposed model outperforms all existing SOTA methods and exhibits good generalization performance on unseen data.
