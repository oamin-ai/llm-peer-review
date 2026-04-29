# Integrating Clinical Knowledge Graphs and Gradient-Based Neural Systems for Enhanced Melanoma Diagnosis via the Seven-Point Checklist

Author: Muhammad Tariq
University: Pennsylvania State University


## Abstract

The seven-point checklist (7PCL) is a widely used diagnostic tool in dermoscopy for identifying malignant melanoma by assigning point values to seven specific attributes. However, the traditional 7PCL is limited to distinguishing between malignant melanoma and melanocytic nevi (MN) and falls short in scenarios where multiple skin diseases with appearances similar to melanoma coexist. To address this limitation, we propose a novel diagnostic framework that integrates a clinical knowledge-based topological graph (CKTG) with a gradient diagnostic strategy featuring a data-driven weighting (GD-DDW) system. The CKTG captures both the internal and external relationships among the 7PCL attributes, while the GD-DDW

Digital Object Identifier 10.1109 / TNNLS.2025.3600443

emulates dermatologists' diagnostic processes, prioritizing visual observation before making predictions. Additionally, we introduce a multimodal feature extraction approach leveraging a dual-attention mechanism to enhance feature extraction through cross-modal interaction and unimodal collaboration. This method incorporates meta-information to uncover interactions between clinical data and image features, ensuring more accurate and robust predictions. Our approach, evaluated on the EDRA dataset, achieved an average AUC of 88.6%, demonstrating superior performance in melanoma detection and feature prediction. This integrated system provides data-driven benchmarks for clinicians, significantly enhancing the precision of melanoma diagnosis.

Index Terms -Data-driven learning, graph convolutional neural networks (CNNs), multilabel, multimodal, skin cancer detection.

## I. INTRODUCTION

S KIN cancer is a widespread malignancy worldwide, with melanoma being the most dangerous type. Although the incidence of skin cancer varies by region and individual risk factors, timely diagnosis and prompt intervention can significantly improve the prognosis of the disease [1].

In the medical field, healthcare professionals utilize diverse imaging modalities to expedite the diagnosis and treatment of skin lesions [2]. Two notable techniques are dermoscopic imaging and clinical photography. Dermoscopy, which is an optical magnification technique assisted by liquid immersion or cross-polarized illumination, produces intricate images revealing subepidermal structures and lesions to support the diagnosis [3]. On the other hand, clinical images e GLYPH&lt;14&gt; ciently capture essential morphological features, including color, texture, shape, and borders, despite their inability to explore intricate subcutaneous lesion details. The convenience and accessibility of clinical images, especially through widely available mobile electronic devices like smartphones, are indisputable benefits [4]. The fusion of these two imaging techniques provides comprehensive information essential in practical medical scenarios, specifically for melanoma diagnosis.

Conventionally, clinical practice relies on pattern analysis, which recognizes dermatologic attributes to aid in skin

lesion diagnosis. Several established algorithms, such as the ABCD rule [5] and the seven-point checklist (7PCL) [6], aid dermatologists in recognizing cancer-related characteristics, contributing to a simpler diagnostic process. For example, this study focuses on the 7PCL algorithm, which evaluates seven dermoscopic characteristics associated with malignant melanoma. These include three major attributes: atypical pigment network (ATP-PN), blue-whitish veil (BWV), and irregular vascular structures (IR-VSs), as well as four minor attributes: irregular pigmentation (IR-PIG), irregular streaks (IR-STRs), irregular dots and globules (IR-DaGs), and regression structures (RSs). Each major attribute receives a weighted score of two points, while minor attributes earn one point each. Cumulative scores provide an initial evaluation, requiring a predetermined threshold-commonly set at one or three points. Scores exceeding this threshold are classified as suspected melanoma, as illustrated in Fig. 1. Although the 7PCL algorithm is widely accepted and utilized, the attributes and their assigned weights are primarily designed to di GLYPH&lt;11&gt; erentiate between malignant melanoma (Mel) and melanocytic Nevi (MN) [7], limiting its range of applications. By analyzing these attributes as key features using more appropriate quantitative criteria, there is potential to broaden its applicability and enhance predictive performance, ultimately benefiting healthcare practitioners.

In this study, we propose a data-driven graph-based framework for melanoma diagnosis that incorporates clinical insights from the 7PCL. Our work is based on the observation that melanoma attributes exhibit directed relationships, influencing each other to varying degrees, as shown in Fig. 2. Additionally, the relative importance of these attributes varies depending on the diagnostic context. In summary, the main contributions of this work are as follows.

- 1) We develop a directed weighted GCN encoding clinically relevant relationships between melanoma attributes, transforming expert knowledge into a structured, data-driven diagnostic system.
- 2) A dual-attention mechanism enables an e GLYPH&lt;11&gt; ective fusion of image features and meta-information, improving feature extraction and predictive performance.
- 3) Our method assigns trainable weights to 7PCL attributes, ensuring adaptability across di GLYPH&lt;11&gt; erent dermatological conditions while preserving the clinical decision logic.
- 4) Our method is evaluated in three open-access datasets, EDRA, ISIC2017, and ISIC2018, demonstrating improved melanoma prediction and attribute recognition.

## II. RELATED WORK

Computer-aided diagnosis (CAD) has rapidly emerged as an e GLYPH&lt;11&gt; ective tool to assist dermatologists in the detection of melanoma, as it can extract characteristic information related to the lesion. The conventional CAD procedure for melanoma diagnosis based on machine learning involves multiple steps. These include image preprocessing (e.g., artifact removal and color correction), lesion segmentation, extraction of handcrafted feature sets, and classification using dedicated classifiers based on the application context [8], [9]. Initially,

Fig. 1. Example of biopsy-confirmed melanoma cases from the EDRA dataset, annotated with attributes from the 7PCL. (a) IR-PIG (1), IR-STR (1), IR-DaG (1), BWV (2), and ATP-PN (2), totaling seven points. (b) IR-DaG (1), IR-VS (2), BWV (2), and IR-PIG (1), totaling six points. (c) IR-DaG (1), ATP-PN (2), IR-PIG (1), and RS (1), totaling five points. All cases exceed the threshold of three points.

<!-- image -->

Celebi et al. [10] introduced a method that automatically detects borders, extracts shape, color, and texture features from relevant regions, selects optimized features, and addresses class imbalance specifically in dermoscopy images of pigmented skin lesions. Based on this work, researchers have proposed ways to take advantage of the variety of handcrafted features of melanoma. Schaefer et al. [11] proposed an ensemble method that combines multiple classifiers and various handcrafted features and demonstrated statistically

Fig. 2. Conditional probability matrix for melanoma and 7PCL attributes. Each cell in the matrix represents the probability of one attribute given the presence of another. The rows correspond to the conditions, and the columns represent the attributes conditioned upon.

<!-- image -->

improved recognition performance. Fabbrocini et al. [12] proposed a framework that integrates handcrafted features and the 7PCL. They employed both machine learning classifiers and statistical analysis to enhance the performance of diagnosis. Wadhawan et al. [9] devised a comparable notion and modified it for intelligent, portable devices aimed at routine clinical skin monitoring or as an aid for dermatologists and primary care physicians in identifying melanoma. These e GLYPH&lt;11&gt; orts have facilitated collaboration between machine learning methods and clinical diagnosis of melanoma. However, their performance has been limited by an overreliance on preprocessing and complicated feature extraction procedures.

Over the past decade, deep learning has become a widely used and powerful tool for feature learning and pattern recognition in CAD, enabling automated detection of melanoma and other skin diseases in various scenarios using di GLYPH&lt;11&gt; erent methods [13], [14], [15], [16]. Esteva et al. [13] demonstrated the classification of skin lesions using a single convolutional neural network (CNN) trained end-to-end directly from both clinical and dermoscopy images, using only pixels and disease labels as input, and achieving performance comparable to dermatologists. Then, Abhishek et al. [15] presented the first work to directly explore image-based predicting clinical management decisions of melanoma without explicitly predicting diagnosis. Meanwhile, the international skin imaging collaboration (ISIC) challenges and the corresponding constantly updated and expanded public datasets have greatly contributed to the rapid development of the field, and more excellent methods and training strategies have been proposed [17]. However, the abovementioned research works focus more on the pixel information brought by the image itself, especially in the imaging modality of dermoscopy. Incorporating expert domain knowledge and multimodal data are important directions that deserve further attention.

Using information from clinical diagnostics to support multimodal deep learning models has proven e GLYPH&lt;11&gt; ective in several related research. Moura et al. [18] combined the ABCD rules and a pretrained CNN through a multilayer perceptron classifier for melanoma detection and achieved improved performance. Kawahara et al. [6] proposed a multitask CNN trained on both dermoscopy and clinical images, along with the corresponding 7PCL attributes for melanoma detection, which has been regarded as the benchmark of the related research studies. Following this work, Bi et al. [19] proposed a hyper-connected CNN (HcCNN) structure for the multimodal analysis of melanoma and the corresponding clinical features. Tang et al. [20] further developed this idea and fused 7PCL attributes information into a deep learning framework in a multistage manner, improving the average diagnostic performance. However, none of these studies have fully considered the intrinsic connections between melanoma attributes recorded as meta-information when analyzing the clinical data. The metainformation is processed consistently as feature vectors in combination with image features, resulting in some improvement in the outcomes. Nevertheless, the potential value of this clinical information warrants further investigation. To address these issues, Graph CNNs (GCNs) are introduced to assist with data mining for works involving supplementary clinical information [21], [22], [23]. Wu et al. [24] introduced GCNs based on general clinical meta-information to aid in the multilabel classification of skin diseases. Wang et al. [25] designed a framework utilizing constrained classifier chains and first examined the mutual relationships between attributes of 7PCL. Fu et al. [22] applied GCNs to a 7PCL-based multimodal and multilabeled classification task and uncovered the interconnections between attributes by analyzing simultaneous occurrences of di GLYPH&lt;11&gt; erent attributes and undirected GCNs. These studies demonstrate the potential of combining graph learning with clinical diagnosis to aid dermatologists in the pre-diagnosis of melanoma. However, a gap persists in understanding the directional relationships among diverse attributes and their practical implications for melanoma diagnosis.

In recent studies, deep learning algorithms incorporating attention mechanisms have gained prominence and achieved significant success in medical image and signal analysis [26], [27], [28], [29]. In the field of dermatology, He et al. [30] introduced a cross-modality structure that applies attention weight maps learned separately from bimodal images to each other, enhancing prediction performance. Xu et al. [31] developed an architecture that integrates global and local attention mechanisms, mimicking the diagnostic process of dermatologists. Additionally, Xiao et al. [32] proposed an attention-guided feature reconstruction method utilizing both clinical and ultrasound images, leveraging intramodality and intermodality information to extract more discriminative features. While these studies have significantly enhanced the performance of deep learning models for dermatological image analysis and improved the interpretability of their results, they primarily focus on interactions among learned image features alone. The integration of meta-information, such as the 7PCL attributes, and its interaction with image features remains underexplored. Furthermore, the potential for AI technology

Fig. 3. Illustration of our proposed melanoma diagnosis framework, featuring a topological graph CNN inspired by clinical knowledge and a gradient diagnosis mechanism. The framework includes (a) multimodal fusion system combining data from dermoscopy and clinical images, and the meta information; (b) generation of graph features from the 7PCL attributes; and (c) gradient diagnostic strategy with a trainable weight module.

<!-- image -->

to provide feedback on the 7PCL and thereby benefit clinical decision-making has yet to be addressed.

## III. METHODOLOGY

As illustrated in Fig. 3, the proposed method comprises three modules. The first module, clinical-dermoscopic multimodal fusion (CD-MFM), combines information from clinical and dermoscopic modalities. The second module, 7PCL directed graph mining (7PCL-DirGM), extracts representative features using a mining strategy that leverages directed and multiorder graph information from multilabeled data. These two modules form our clinical knowledge-based topological graph (CKTG) convolutional network. Additionally, the gradient diagnostic strategy featuring a data-driven weighting (GD-DDW) employs data-driven weighted gradient diagnosis to enhance diagnostic performance by emulating a dermatologist's diagnostic behavior. Sections III-A-III-C provide a detailed overview of the methodology.

## A. Clinical-Dermoscopic Multimodal Fusion

In this study, we treated each skin lesion as an individual 'case' within the dataset. Each case, denoted as x i , encompasses data from multiple modalities, including dermoscopy images x i d , clinical images x i c , and the encoded information related to the j th attribute within the 7PCL, x i 7p j , where i ranges from 1 to n (the total number of cases) and j ranges from 1 to 7. Additionally, each case is associated with a diagnostic tag, denoted as y i m , which represents the melanoma diagnosis, and labels y i 7p j for their 7PCL attributes.

First, residual convolutional blocks are applied to independently extract visual features from dermoscopic and clinical images, denoted as X d and X c, respectively. Simultaneously, a BERT-based semantic embedding layer is used to encode the label information into semantic features X m, which represent the diagnosis and 7PCL attributes. These semantic features are aligned with the visual features from the two image modalities. Next, two separate concatenation layers are applied to combine the visual features and the semantic features, resulting in X dm for the dermoscopic modality and X cm for the clinical modality. These combined features are then passed through dual-attention feature extraction blocks for further processing.

As shown in Fig. 4, the dual-attention feature extraction blocks consist of both single-modality attention (SMA) [33] and cross-modality attention (CMA) blocks [30]. The features X dm and X cm are first refined by their respective SMA blocks, which capture intramodality interactions within the dermoscopic and clinical image embeddings, embedding D and embedding C, respectively, along with their meta-information in embedding m. These features then pass through residual blocks before being processed by the CMA block, which facilitates cross-modality interaction between the dermoscopic and clinical features. This enables the integration of complementary information from both modalities, enhancing the joint feature representation.

In detail, as shown in Fig. 5, the input features X dm and X cm are transformed into query ( Q ), key ( K ), and value ( V )

Fig. 4. Illustration of the dual-attention blocks.

<!-- image -->

Fig. 5. Illustration of (a) SMA and (b) CMA blocks. The operator GLYPH&lt;8&gt; denotes element-wise addition, while GLYPH&lt;10&gt; represents element-wise multiplication.

<!-- image -->

representations for each modality. The attention scores for each modality are then computed as

<!-- formula-not-decoded -->

The refined features for each modality are then obtained by applying the attention scores to the corresponding value representations in the SMA block

<!-- formula-not-decoded -->

Similarly, the CMA block facilitates interaction between the two modalities by sharing attention across them. The crossmodality refined features are computed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where X CMA GLYPH&lt;0&gt; dm represents the dermoscopic features refined using clinical information and X CMA GLYPH&lt;0&gt; cm represents the clinical features refined using dermoscopic information. By leveraging both intra- and CMA, the model ensures that relevant information from both dermoscopic and clinical images is fully captured and integrated for a more robust melanoma diagnosis.

Next, as shown in Fig. 3, after the feature extraction process through the four dual-attention blocks, we obtain the refined feature embeddings X d GLYPH&lt;0&gt; refined and X c GLYPH&lt;0&gt; refined for the individual image modalities, which are combined with meta-information extracted via the SMA module. Additionally, the fusion feature X f GLYPH&lt;0&gt; refined is obtained through the CMA module. These three components are then prepared for further fusion with the information provided by the graph neural network 7PCLDirGM.

Let Z represent the features extracted from the graph information. In the next fusion step, we individually combine the graph features ( Z ) with the deep features using elementwise multiplication

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where X d GLYPH&lt;0&gt; fused , X c GLYPH&lt;0&gt; fused, and X f GLYPH&lt;0&gt; fused represent the fused features for the dermoscopic, clinical, and combined channels, respectively.

Subsequently, we apply a fully connected layer to each channel independently and perform a weighted average fusion of the features

<!-- formula-not-decoded -->

where GLYPH&lt;13&gt; d, GLYPH&lt;13&gt; c , and GLYPH&lt;13&gt; f represent the weights for combining the fully connected layers of the dermoscopic, clinical, and additional channel images, respectively.

The resulting fused feature, X , with dimensions R n × 768 , is utilized for downstream tasks, where n represents the number of cases and 768 corresponds to the feature dimensionality, aligned with the meta-embedding layer. This process e GLYPH&lt;11&gt; ectively integrates information from both the 7PCL graph and the image modalities.

## B. 7PCL Directed Graph Mining

Within the 7PCL dataset, there exist both directional and causal relationships among the various attributes, each associated with di GLYPH&lt;11&gt; erent stages of melanoma development and their causes. For instance, both irregular pigment network and atypical pigmentation are important features to consider when evaluating potential melanomas. However, an irregular pigment network is often considered a more specific and reliable indicator of melanoma when compared to atypical

pigmentation alone. This is because the disruption of the normal pigment network structure is a hallmark feature of melanomas and is less commonly seen in benign moles. Therefore, the presence of an irregular pigment network raises the suspicion for melanoma and may prompt further evaluation or biopsy [34], [35], [36]. Recognizing these distinctions aids in filtering out superfluous and inaccurate interactions, enhancing the e GLYPH&lt;14&gt; ciency of graph network data utilization. Moreover, contrary to previous studies equating disease diagnosis labels with 7PCL features [6], [19], [20], [22], we emphasize that the information pertaining to disease diagnosis, the various attributes, and interactions among these attributes represent significant directed information that should be considered separately with appropriate weighting for e GLYPH&lt;11&gt; ective integration. To address these complexities, our study proposes the 7PCLDirGM, a directional graph mining method tailored to extract vital attribute interaction data related to melanoma.

1) Graph Node Feature Encoding: In graph convolutional learning networks, the encoding of node information-often critical-tends to be overshadowed by the emphasis on topological relationships between nodes. Previous research frequently relies on one-hot encoding to represent textual information features; however, this approach reduces the richness of node attributes to simple labels, limiting the depth of information available for learning. For instance, positive expressions related to attributes like pigmentation and streaks often include irregular descriptive terms, which carry significant semantic information.

In this study, we employ the BERT embedding strategy [37], which not only preserves semantic context but also dynamically captures node feature details. Each node feature, denoted as x i 7p j , is transformed into an embedding X i 7p j 2 R 7 × 768 using the BERT module, where 768 is used to match the default output dimensionality of BERT. Here, 768 was chosen based on pre-experiments to optimize the representation of encoded features.

2) Internal and External Directional Weighted Graph (IEDWG): One significant deviation from related studies is the importance we place on the directed connection between attributes in 7PCL for constructing directed graph networks. As introduced in Algorithm 1 and Fig. 6, this connection is reflected primarily in two levels: the internal level within the attributes, which is determined by the mutual connections within the seven attributes in the checklist. The external level involves the coexistence probability of each attribute and the melanoma. Both internal and external information are used to build the weighted directional edges by using the conditional coexistence probability directedness wp ; q , which is calculated based on the training data. This probability serves as the weight of all edges and illustrates the unequal dependency relationship among attributes, as shown in Fig. 7.

3) Adaptive Receptive Field Proximity: Accurately modeling attribute relationships is crucial for melanoma diagnosis using the 7PCL. However, conventional graph-based methods struggle to capture long-range dependencies, leading to imbalanced receptive fields and suboptimal feature learning. To address this, we introduce adaptive receptive field proximity (ARFP), which enhances the multiorder connectivity of the

<!-- image -->

7PCL-directed weighted graph, improving melanoma classification. Inspired by digraph inception convolutional networks [38], ARFP leverages the k th-order proximity to incorporate both direct and indirect attribute interactions. Given a graph G = ( V ; E ), where V represents 7PCL attributes and E represents their directed relationships, the proximity paths are defined as

<!-- formula-not-decoded -->

where M ( k ) p ; q and D ( k ) p ; q represent the meeting and di GLYPH&lt;11&gt; usion paths between attributes p and q ( p ; q 2 V ). If both paths exist, the attributes are at the k th-order proximity level, with ve as their common neighbor.

Using the 7PCL-directed weighted graph and the k th-order proximity, the 7PCL k th-order proximity matrix founded on the conventional spectral-based GCN method is denoted in

Fig. 6. Illustration depicts the principles of both internal and external connectivity. The green nodes and matrix represent external connections between the seven attributes and melanoma, while the blue nodes and matrix represent internal connections among the seven attributes themselves. The gray matrix represents the final matrix obtained by the weighted fusion of the two. (a) External connectivity. (b) Internal connectivity. (c) Combined connectivity.

<!-- image -->

Fig. 7. Plot illustrates the weighted connections in the training set of the EDRA dataset. (a) Focus on a specific pair, ATP-VS and IR-PIG. Yellow lines represent connections from ATP-VS to the other nodes, while cyan lines represent connections from IR-PIG. Variations in line thickness highlight di GLYPH&lt;11&gt; erences in directed weights between the nodes. (b) Overall connectivity map among all eight nodes.

<!-- image -->

(7), as shown at the bottom of the page, where A represents the adjacency matrix of the directed weighted graph G and D represents the corresponding diagonalized degree matrix. The matrix intersection operation intersect( GLYPH&lt;1&gt; ) creates a new matrix with the element-wise intersection of both meeting and di GLYPH&lt;11&gt; usion paths. This operation aims to symmetrize the k thorder proximity matrix P ( k ) to fit the structure of spectralbased GCN. Here, the value of parameter k represents the distance between two attributes, which determines the size of the receptive fields. By adjusting the value of parameter k , users can observe and select the optimal receptive field with scalable capabilities.

Based on the k th-order proximity matrix, the multiscale digraph is then defined as

<!-- formula-not-decoded -->

where Z ( k ) represents the convolved output with dimensions R n × d and X denotes the node feature matrix with dimensions R n × c . W ( k ) refers to the weight matrix, which is diagonalized from P ( k ) , and GLYPH&lt;2&gt; ( k ) , with dimensions R c × d , is a trainable weight matrix. It is noteworthy that when k = 1, Z (1) is computed through digraph convolution using P (1) as defined in (7), and GLYPH&lt;5&gt; (1) represents the corresponding approximate diagonalized eigenvector associated with P (1) .

This approach enables a scalable representation of 7PCL attribute relationships, ensuring that clinically meaningful dependencies are preserved and e GLYPH&lt;11&gt; ectively utilized in melanoma diagnosis. By leveraging the adaptive receptive field, our framework refines multiscale learning across directed attributes, improving classification performance and interpretability.

## C. Gradient Diagnostics With Data-Driven Weighting (GD-DDW)

As previously discussed, the deep features extracted from fused imaging modalities and the directed mutual connection matrix are collectively referred to as X for subsequent diagnostic procedures. In this section, we present the GDDDW method, designed to emulate the diagnostic approach employed by dermatologists using the 7PCL algorithm in clinical practice. Specifically, the GD-DDW method utilizes representative features corresponding to various attributes to make initial predictions, which are then used to diagnose melanoma.

In this module, we first establish a parallel multilabel classifier designed to address the classification of seven attributes on the checklist. As X is the input, y 7p j = [ y 7p 1 ; y 7p 2 ; : : : ; y 7p 7 ] 2 f 0 ; 1 g 7 is denoted as the ground truth of all seven attributes in the format of a binary indicator vector. In the proposed method, the focal loss function [39] was employed with the objective of enhancing the performance on the dataset that exhibits a serious imbalance.

<!-- formula-not-decoded -->

The focal loss function for each label can be defined as

<!-- formula-not-decoded -->

where ˆ y 7p j represents the predicted probability of the positive class, GLYPH&lt;22&gt; j is a balancing parameter, and GLYPH&lt;28&gt; is the focusing parameter. The total focal loss L 7p for all seven attributes can be formulated as

<!-- formula-not-decoded -->

Next, the prediction scores obtained for each label are utilized as inputs for the subsequent step. In this stage, a weighted sum function and a sigmoid activation function are applied to perform classification for the diagnosis of melanoma.

Let w = [ w 1 ; w 2 ; : : : ; wj ] represent the learned weights module obtained from the attributes. In this module, the predicted probability of melanoma ˆ y m is computed as

<!-- formula-not-decoded -->

where GLYPH&lt;27&gt; ( GLYPH&lt;1&gt; ) denotes the sigmoid activation function and the rescaling factor GLYPH&lt;16&gt; 1 = P 7 j = 1 wj GLYPH&lt;17&gt; ensures that the output is appropriately scaled to the range [0 ; 1].

The overall loss L is then computed as the sum of L 7p and L mel

<!-- formula-not-decoded -->

where GLYPH&lt;21&gt; is the weighting parameter that adjusts the contributions of the two losses, specifically, making sure that the loss values are in the same order of magnitude.

## IV. EXPERIMENTS

## A. Materials

In this study, we utilized the publicly available EDRA dataset (also known as Derm7pt), specifically curated for 7PCL studies and annotated by Kawahara et al. [6]. This dataset comprises paired dermoscopic and clinical images sourced from 1011 patients; each image has a maximum resolution of 768 × 512 pixels. Notably, nine patients lacked clinical images, which were substituted with corresponding dermoscopic ones. Throughout our study, we strictly adhered to the dataset's o GLYPH&lt;14&gt; cial specifications, which stipulated a total of 413 training, 203 validation, and 395 test images. Concerning the labels, the 7PCL attributes were incorporated into the analysis, following the o GLYPH&lt;14&gt; cial categorization. It is worth noting that the original disease classification (DIAG) system included specific disease labels for MISC, BCC, Mel, SK, and Nevi. This study primarily focused on distinguishing melanoma from lookalike diseases. To verify the generalizability of our proposed method, we also used two publicly available datasets, ISIC 2017 and ISIC 2018 [17], which also have multiple attributes. The visual attributes contained in ISIC 2017 include milia-like cysts, negative network, pigment network, and streaks. A total of 2000 training, 150 validation, and 600 test images. ISIC 2018 adds an additional attribute, globules, to the 2017 version and adds a portion of the data to make the data distribution adjusted to 2594 training, 100 validation, and 1000 test images.

## B. Implementation Details

In our study, we ensured the integrity of our results by employing consistent data processing methods and adhering to uniform model training protocols across various ablation comparison experiments. Our chosen deep learning model framework is based on three residual blocks. The entire architecture is meticulously fine-tuned end-to-end, utilizing the Adam optimizer [40] with a concurrent learning rate set to 0.00001. We predefined the number of epochs for pretraining to 150, with an early stopping mechanism in place. If the model's performance on the validation set fails to improve over 50 consecutive epochs, the training process halts, and the best-performing model is saved. The loss functions utilized are based on the focal loss to circumvent the result bias that may arise from imbalanced data. The hyperparameters for the di GLYPH&lt;11&gt; erent stages outlined in the methodology were selected based on insights gained from the training process and represent approximate ranges. The initial stage of the process is the preprocessing of the images. To ensure consistency and minimize the impact of varying image formats, we removed black edges from the images and resized them to 512 × 512 pixels, maximizing the use of image features while avoiding information loss.

During the data preparation phase, all labels are transformed into standardized textual descriptions. These are then processed by the semantic embedding function to e GLYPH&lt;11&gt; ectively utilize label information. Each patient case follows the format: 'This patient has been diagnosed with [Diagnosis], exhibiting the attributes [Attribute 1], [Attribute 2], and [Attribute 3].' This structured representation ensures consistency in data processing and facilitates the meaningful integration of clinical attributes into the model. The standard texts of the indicated 7PCL are ATP-PN, IR-STRs, IR-PIG, RS, IR-DaGs, BWV, and IR-VSs.

In the CD-MFM section, features from di GLYPH&lt;11&gt; erent channels undergo two stages of fusion using weighted averaging. Initially, a fusion process occurs between clinical and dermoscopy images using the dual-attention blocks, followed by a second fusion stage after incorporating topological relations learned by the graph network. The parameters GLYPH&lt;13&gt; series in (5) are configured to 1 = 3 to mitigate potential biases stemming from assigning varying weights to the same feature across multiple instances. In constructing the graph network, we consider only a subset of the possible connections between nodes due to the directionality of the multistep connections. Based on the average number of coexisting labels in the training set and the avoidance of isolated points, the connections are ordered according to the edge weights, ensuring that each node has at least one and no more than three edges. The order of connection k is set to 3 due to the natural limitation of the number of nodes in 7PCL. In the final diagnostic phase, the parameter GLYPH&lt;21&gt; , utilized in the global optimization of the loss function in (12), is set to 1, given that both loss values are

TABLE I CLASSIFICATION PERFORMANCE (AUC, SENS, AND SPEC) OF THE PROPOSED METHOD COMPARED WITH THE SOTA METHODS, EVALUATED ON THE

EDRA DATASET. TOP TWO AUC SCORES PER LABEL ARE HIGHLIGHTED

of a comparable magnitude. All experiments were conducted on one NVIDIA Tesla V100 GPU card (32-GB memory). The duration of each training cycle is approximately 1 h.

## C. Evaluation Settings

The study used standard performance metrics such as AUC, sensitivity, and specificity. To demonstrate our method's advantages, we compared it with state-of-the-art (SOTA) methods like RemixFormer ++ [31], CAFNet [30], GRM-DC [22], 7PCL-Constrained-CC [25], HcCNN [19], TripleNet [41], CCRD [42], GELN [43], and EmbeddingNet [44]. Specifically, using the Inception models introduced by Kawahara et al. [6] as the baseline for comparison, as they originally published the data along with the method.

We also conducted ablation experiments to evaluate the critical components of our method. These experiments included comparisons between unimodal and multimodal fusion (dualattention), assessments of our directional graph mining-based GCN strategy (7PCL-DirGM) and tests of our gradientdriven prediction structure (GD-DDW) against parallel architectures.

In addition, the prediction results for all attributes obtained from the first layer of the GD-DDW structure were utilized. Following the traditional 7PCL rule, diagnostic scores were calculated by summing the assigned attribute scores. Thresholds ranging from 1 to 7 were applied to diagnose melanoma. These results were then compared with the proposed method across two tasks: Mel versus Nevi and Mel versus other diseases, including MISC, SK, Nevi, and BCC.

We further validated our method using two additional publicly available datasets, ISIC 2017 and ISIC 2018. Although these datasets provide only attribute information and lack diagnostic labels, preventing the evaluation of GD-DDW's impact, they still allow the validation of directed multiorder relationships between the attributes (CKTG).

## V. RESULTS

## A. Comparison to SOTA Methods

Table I presents a detailed comparison of the classification performance between our proposed method and SOTA deep learning-based techniques on the EDRA dataset. In this table, the top two AUC values for each label have been bolded to emphasize the highest-performing methods. Our approach achieved an average AUC of 88.6%, demonstrating strong generalization and robustness across multiple diagnostic labels. Specifically, it attained the highest AUC for four of the eight diagnostic labels: STR (90.1%), PIG (88.3%), RS (89.3%), and DaG (86.2%). For the remaining labels, it consistently ranked among the top two, underscoring its reliability across di GLYPH&lt;11&gt; erent classification tasks.

Building on this, when analyzing the comparative performance of existing models, all methods evaluated in Table I are deep learning-based. However, only GRM-DC and GELN incorporate GCNs, which explicitly model relationships

TABLE II COMPARATIVE ABLATION STUDY OF KEY MODULES IN OUR PROPOSED METHOD, EVALUATED ON AUC USING THE EDRA DATASET

between clinical attributes. Our approach further advances this direction by introducing the consideration of directed interactions between attributes and melanoma, leading to improved experimental results. These findings indicate that modeling directed relationships enhances diagnostic performance and contributes to a more interpretable model.

As the most recent SOTA method aside from our proposed approach, RemixFormer ++ integrates a transformer-based feature extractor with clinical diagnostic logic, achieving a strong performance. In contrast, our method further embeds clinical rules into the graph network while incorporating the self-attention mechanism used in transformers to enable multimodal task learning. This design reduces model size while enhancing performance. In the eight-label classification task, our approach achieved improvements in six categories, increasing the average AUC from 86.3% to 88.6%.

Although our proposed method achieves improvements in both AUC and sensitivity, its advantage over other methods in specificity (85.6%) is not as pronounced. This is primarily due to the detection mechanism of the 7PCL clinical algorithm, which is designed to maximize the identification of potential melanoma cases and ensure that positive patients are not overlooked. As a result, it inherently favors high sensitivity over specificity [45]. In our proposed method, we address this issue by optimizing the weight distribution and threshold selection, enabling the model to maintain a high sensitivity (69.8%) while achieving competitive specificity. This balanced approach enhances the reliability of the results and makes the model more clinically applicable.

## B. Ablation Studies

The proposed methodological framework consists of multiple modules, each serving a distinct role: CD-MFM facilitates comprehensive multimodal feature learning, 7PCL-DirGM enhances graph-based representation learning with directed interactions, and GD-DDW integrates clinical diagnostic knowledge to improve interpretability and decision-making. To systematically evaluate the contribution of each module, Table II presents a comparative analysis through an ablation study.

The first comparison assesses unimodal inputs, where dermoscopic and clinical images are processed separately, against multimodal fusion using dual-attention blocks. The results indicate that integrating dermoscopic and clinical imaging via dual-attention mechanisms improves the average AUC by 7%-10%, highlighting the benefits of leveraging complementary information from di GLYPH&lt;11&gt; erent imaging modalities. Additionally, dermoscopic images alone outperform clinical images by an average of 3%, which aligns with expectations since dermoscopy provides enhanced feature visibility, particularly in the context of attribute-based analysis within the 7PCL framework.

Further analysis examines the individual impact of the 7PCL-DirGM and GD-DDW modules. When incorporated independently, both modules contribute to an average AUC improvement of 9% over the baseline dermoscopic model, with a particularly notable 13% increase in melanoma classification. These findings emphasize the critical role of incorporating structured graph-based relationships and gradient-based diagnostic decision weighting, both of which enhance the model's ability to capture clinically relevant feature interactions.

The final evaluation considers the combined integration of all three modules, demonstrating a synergistic e GLYPH&lt;11&gt; ect that further elevates model performance. The full framework achieves an average AUC of 87%, representing a significant enhancement over individual components. These results validate the e GLYPH&lt;11&gt; ectiveness of combining multimodal feature learning, directed graph modeling, and diagnostic decision weighting, reinforcing the importance of clinically guided deep learning approaches for melanoma diagnosis.

## C. Comparison to Traditional 7PCL Algorithm

The approach presented in this article is primarily inspired by the 7PCL clinical scoring mechanism for melanoma diagnosis, with key advancements introduced through dynamic weight assignment and automated diagnosis enabled by GD-DDW. These improvements not only enhance diagnostic performance and expand the applicability of the model but also generate data-driven weighting information that can provide valuable feedback to frontline clinicians. By leveraging

TABLE III COMPARISON OF WEIGHT VALUES BETWEEN THE PROPOSED METHOD AND TRADITIONAL ALGORITHM UNDER DIFFERENT TASKS

Fig. 8. ROC comparison graph showcasing the performance di GLYPH&lt;11&gt; erences between the proposed method and the traditional 7PCL algorithm across two classification tasks. (a) Classification of Mel and Nevi. (b) Classification of Mel and other diseases (MISC, Nevi, BCC, and SK).

<!-- image -->

real-world data for adaptive weighting, the proposed method addresses the inherent subjectivity in predefined scoring systems and improves the model's interpretability.

To systematically evaluate these improvements, we compare the performance of the proposed method with the traditional 7PCL approach in two distinct diagnostic scenarios. The first follows the conventional application of 7PCL, distinguishing between Mel and Nevi, where the original scoring system has been widely validated. The second considers a more complex diagnostic setting in which Mel must be di GLYPH&lt;11&gt; erentiated from a broader range of skin diseases. This latter scenario reflects the challenges encountered in real-world clinical applications, where dermatological conditions exhibit significant variability. By assessing performance across these scenarios, we aim to highlight the benefits of adaptive learning and dynamic weight assignment in improving diagnostic performance beyond traditional rule-based approaches.

As shown in Table III, the traditional algorithm employs approximate categorization based on multivariate analysis [7], assigning weights of 2 or 1 to major and minor attributes, respectively. In contrast, the method proposed in this study automatically generates adaptive weights based on the prediction scores of each category under di GLYPH&lt;11&gt; erent tasks. Specifically, in the classification task of di GLYPH&lt;11&gt; erentiating Mel from Nevi, the weights of the major attributes range from 1.7 to 2.4, while the weights of the minor attributes range from 0.8 to 1.4, closely resembling traditional algorithms. However, when the task shifts to distinguishing Mel from four other skin diseases, the weights dynamically adjust to range from 1.3 to 1.5 for major attributes and from 0.9 to 1.1 for minor attributes.

This dynamic adjustment extends the applicability of the 7PCL algorithm, originally designed to distinguish Mel from Nevi, beyond its initial scope. As shown in Fig. 8(b), the traditional algorithm experiences a significant drop in AUC, from 92% to 72%, when applied to more complex scenarios.

TABLE IV PERFORMANCE (AUC) OF THE PROPOSED METHOD ON ISIC DATASETS

In contrast, our proposed data-driven dynamic tuning approach maintains a high AUC value of 92% while using the same features in the classification tasks of Mel versus Nevi and Mel versus others. It should be noted that the data-driven weight parameters in Table III are derived from model training and can be dynamically optimized as more data become available. These results indicate that although these attributes were originally designed for distinguishing Mel from Nevi, their diagnostic relevance extends beyond this binary classification. By enabling automated dynamic weight adjustment, the proposed method e GLYPH&lt;11&gt; ectively preserves the value of these attributes in di GLYPH&lt;11&gt; erentiating Mel from a broader range of skin diseases. This adaptability highlights the potential of leveraging clinically established features while optimizing their contribution based on task-specific requirements, thereby improving the model's generalization in more complex diagnostic scenarios.

## D. Generalizability Analysis on the ISIC Datasets

Generalizability is a key factor in assessing a method's clinical applicability. To evaluate this, we tested our approach on the ISIC 2017 and ISIC 2018 datasets and compared it with SOTA methods. While these datasets contain only dermoscopic images, they encompass a diverse range of skin conditions, including melanoma, Nevi, and SK, and provide attribute annotations beyond the 7PCL framework, such as milia-like cysts. This evaluation assesses the model's ability to generalize beyond 7PCL-specific attributes while accounting for dataset variations, demonstrating its robustness and adaptability to real-world dermatological diagnosis.

As shown in Table IV, we compared the predictive performance of our method against the baseline model (ResNet-50) and SOTA approaches (GRM-DC and RemixFormer ++ ). Our method achieved the highest average AUC across both datasets, with 87.3% on ISIC 2017 and 86.4% on ISIC 2018. It also attained the highest classification performance in six out of nine attributes, further demonstrating its robustness. These results highlight the e GLYPH&lt;11&gt; ectiveness of our method in extracting representative features and modeling attribute interactions, enhancing predictive performance for broader dermatological applications.

## VI. DISCUSSION

In this study, we proposed a novel approach to melanoma diagnosis, integrating clinical empirical knowledge, diagnostic

TABLE V COMPARISON OF DIFFERENT SEMANTIC EMBEDDING METHODS

procedures, and GCNs. Specifically, we improved the causal learning of melanoma-associated attributes to construct an optimized topological graph structure. By incorporating computational logic from clinical algorithms, our method enhances interpretability and clinical acceptance by strengthening the correlation between diagnostic outcomes and imaging datasets. Additionally, several key aspects identified during this study warrant further discussion.

## A. Multiorder Node Associations and Their Impact

Our results demonstrate that multiorder node associations, introduced through ARFP, play a crucial role in melanoma detection performance. Beyond the main results presented in Section V, additional experimental evaluations reveal a consistent improvement in AUC as the order k increases: 87.1% at k = 1, 90.6% at k = 2, and 92.0% at k = 3, indicating that expanding the receptive field e GLYPH&lt;11&gt; ectively captures more informative attribute relationships. However, at k = 4, AUC slightly declines to 91.8%, suggesting a performance plateau. Given that the 7PCL graph comprises only seven nodes, further increasing k reduces the number of meaningful paths, leading to information dilution rather than enrichment.

These observations highlight the importance of selecting an optimal order k to balance local feature aggregation and global information propagation. Future work may explore dynamic receptive field adjustment or adaptive graph construction based on dataset-specific connectivity patterns. Additionally, incorporating more informative labels could help mitigate the limitations imposed by the small number of nodes, potentially enhancing graph-based melanoma classification performance.

## B. Semantic Information and Its Role in Graph Learning

Existing research on graph-based melanoma detection has primarily focused on link relationships between nodes, while the semantic content of node labels has often been overlooked [22], [38], [43]. To the best of our knowledge, this study is the first to integrate dynamic semantic information into both feature extraction and the graph learning process, enabling full utilization of diagnostic knowledge.

As shown in Table V, we evaluated multiple semantic embedding methods, including one-hot encoding [46], GloVe [47], Word2Vec [48], and BERT [37]. Each method di GLYPH&lt;11&gt; ers in its ability to capture contextual relationships among node labels, with dynamic embeddings o GLYPH&lt;11&gt; ering greater adaptability.

Results indicate that models incorporating semantic embeddings outperform those without them, with BERT achieving the highest AUC values of 92.0% for melanoma classification and an average AUC of 88.6% across all attributes. However, the inclusion of dynamic embeddings comes at the cost of increased computational time, particularly with BERT requiring 84 min of training time compared to 56-63 min for static embeddings. Despite this tradeo GLYPH&lt;11&gt; , the improved diagnostic performance demonstrates the e GLYPH&lt;11&gt; ectiveness of integrating contextual knowledge into graph-based melanoma detection.

These findings suggest that leveraging dynamic embeddings allows the model to capture high-order semantic relationships, ultimately enhancing classification performance. Future research could explore hybrid approaches, balancing computational e GLYPH&lt;14&gt; ciency with embedding e GLYPH&lt;11&gt; ectiveness for large-scale dermatological datasets.

## C. Clinical Implications and Adaptive Weighting for 7PCL

From a clinical perspective, the 7PCL algorithm was originally developed to assist dermatologists in distinguishing Mel from Nevi based on dermoscopic features. However, adapting it into a modern CAD system that encompasses a broader range of similar-looking skin lesions and integrates both clinical and dermoscopic features presents significant challenges. One key limitation is that the original scoring system, which assigns a weight of 2 to major features and 1 to minor features, was designed specifically for Mel and Nevi di GLYPH&lt;11&gt; erentiation and may not generalize well to more diverse diagnostic scenarios [7], [49], [50].

To address these limitations, this study introduces a datadriven approach that dynamically adjusts attribute weights based on the classification task. By refining the weighting mechanism, the 7PCL algorithm can adapt to di GLYPH&lt;11&gt; erent dermatological conditions while maintaining its clinical interpretability. This adjustment allows the system to better align with real-world diagnostic needs, making it more applicable in various clinical settings and improving its utility as a decisionsupport tool for medical professionals.

## D. Potential for Integrating Foundation Models

Foundation models, particularly large-scale multimodal architectures, have demonstrated remarkable adaptability across various domains, including dermatology. These models, pretrained on vast and diverse datasets, o GLYPH&lt;11&gt; er strong feature representations that can be fine-tuned for specific medical tasks, including melanoma diagnosis. Given their ability to capture complex hierarchical patterns and cross-modal interactions, integrating foundation models into our framework could further enhance its robustness and generalization. Specifically, vision-language models like SkinGPT-4 and pathology foundation models have shown promise in leveraging both textual clinical knowledge and visual cues, potentially enriching the diagnostic reasoning process in automated systems [51], [52], [53], [54].

Incorporating foundation models into our approach could provide several advantages. First, they could refine the dynamic weighting mechanism by learning from broader datasets, improving the adaptability of the 7PCL-based scoring system beyond its original scope. Second, their self-supervised

learning capabilities could mitigate the limitations of annotated medical datasets, allowing for improved feature extraction even in data-constrained environments. Finally, by integrating pretrained multimodal embeddings, our framework could benefit from enhanced attribute interactions and context-aware diagnostics, making it more interpretable and clinically applicable. Future work will explore the feasibility of integrating foundation models to further advance automated melanoma diagnosis.

## VII. CONCLUSION

In this study, we present an artificial intelligence-enhanced 7PCL method for melanoma detection, designed to integrate a clinically-informed topology that mirrors the diagnostic approach of dermatologists. This approach addresses the limitations of the traditional 7PCL, which has been primarily restricted to di GLYPH&lt;11&gt; erentiating Mel from Nevi, by enabling automated adaptation to diverse diagnostic scenarios. Our method e GLYPH&lt;14&gt; ciently extracts visual features using an architecture that combines single-modal and cross-modal attention mechanisms, capturing subtle relationships within and across modalities. By incorporating directed multiorder relationships and gradient prediction structures, our approach dynamically assigns weighted metrics to each attribute. This allows the algorithm to adapt to varying diagnostic tasks, such as distinguishing Mel from a broader range of skin conditions with overlapping visual characteristics. This enhanced weighting system not only improves detection performance but also provides dermatologists with a more intuitive and interpretable framework for attribute analysis. By aligning the algorithm's design with clinical workflows, the proposed method could enhance both the e GLYPH&lt;11&gt; ectiveness and acceptance of AI-assisted diagnosis in real-world settings.
