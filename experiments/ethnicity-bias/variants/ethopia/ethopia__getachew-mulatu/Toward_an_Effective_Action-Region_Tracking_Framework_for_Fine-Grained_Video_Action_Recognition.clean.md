# Toward an Effective Action-Region Tracking Framework for Fine-Grained Video Action Recognition

Author: Getachew Mulatu
University: Pennsylvania State University


## Abstract

Fine-grained action recognition (FGAR) aims to identify subtle and distinctive di GLYPH&lt;11&gt; erences among fine-grained action categories. However, current recognition methods often capture coarse-grained motion patterns but struggle to identify subtle details in local regions evolving over time. In this work, we introduce the action-region tracking (ART) framework, a novel solution leveraging a query-response mechanism to discover and track the dynamics of distinctive local details, enabling distinguishing similar actions e GLYPH&lt;11&gt; ectively. Specifically, we propose a region-specific semantic activation module that employs discriminative and text-constrained semantics serve as queries to capture the most action-related region responses in each video frame, facilitating interaction among spatial and temporal dimensions with corresponding video features. The captured region responses are then organized into action tracklets, which characterize the region-based action dynamics by linking related responses across di GLYPH&lt;11&gt; erent video frames in a coherent sequence. The textconstrained queries are designed to expressly encode nuanced semantic representations derived from the textual descriptions of action labels, as extracted by the language branches within visual language models. To optimize generated action tracklets, we design a multilevel tracklet contrastive constraint among multiple region responses at spatial and temporal levels, which can e GLYPH&lt;11&gt; ectively distinguish individual region responses in each video frame (spatial level) and establish the correlation of similar region responses between adjacent video frames (temporal level). In addition, we implement a task-specific fine-tuning mechanism to refine textual semantics during training. This ensures that the semantic representations encoded by vision language models (VLMs) are not only preserved but also optimized for specific task preferences. Comprehensive experiments on several widely used action recognition benchmarks, i.e., FineGym, Diving48, NTURGB-D, Kinetics, and Something-Something, clearly demonstrate the superiority to previous state-of-the-art baselines.

Index Terms -Action recognition, action-region tracking (ART), action tracklet, fine-grained, text-constrained queries.

## I. INTRODUCTION

F INE-GRAINED action recognition (FGAR) is aimed at distinguishing subtle and discriminative di GLYPH&lt;11&gt; erences within fine-grained action categories, aligning more closely

Digital Object Identifier 10.1109 / TNNLS.2025.3602089

with the increasing complexity of human activities in the real world. This emerging area has driven the evolution of action recognition tasks toward a finer granularity, attracting significant attention within the research community due to its potential to enhance various visual analytics applications, including intelligent surveillance [1], social scene understanding [2], and sports video analysis [3].

Recently, video action recognition research has advanced significantly, driven by the emergence of powerful model architectures [4], [5], [6], [7], [8], [9], [10] and the availability of large-scale datasets [11], [12], [13], [14]. The inherent challenge of video action recognition lies in addressing how to e GLYPH&lt;11&gt; ectively encode the spatiotemporal representations. Existing methods can be broadly categorized into two groups: spatiotemporal feature modeling methods (e.g., TSM [15], ATM [16], and Uniformer [17]) and long-temporal modeling methods (e.g., SlowFast [18], TDN [19], and MViT [20]). For example, TSM [15] enhances motion pattern representation by integrating spatiotemporal features with feature-level motion encoding, allowing for a more dynamic capture of motion. Uniformer [17] proposes to e GLYPH&lt;11&gt; ectively combine 3-D convolution with spatiotemporal self-attention mechanisms within a streamlined Transformer architecture, thereby achieving a preferable balance between e GLYPH&lt;14&gt; ciency and e GLYPH&lt;11&gt; ectiveness. TDN [19] addresses multiscale temporal dynamics by di GLYPH&lt;11&gt; erentiating and integrating temporal features across short-term and long-term intervals. MViT [20] introduces a multiview Transformer architecture for video recognition that employs distinct encoders for processing various temporal segments, with lateral connections to fuse information across views.

In previous general action recognition methods, visual appearance cues of objects and background often play a crucial role, sometimes even more significant than the action itself. For example, in distinguishing between actions like ' basketball ' and ' gymnastics ,' the scene information alone can easily allow for di GLYPH&lt;11&gt; erentiation. FGAC is significantly more challenging than general action recognition, primarily due to subtle interclass di GLYPH&lt;11&gt; erences. These di GLYPH&lt;11&gt; erences are often more pronounced in the temporal dimension, where motion information are dominant over visual appearance. Specifically, key motion di GLYPH&lt;11&gt; erences are observed in targeted regions of the human body, including the pose adjustments, the extent of movement range, and the dynamic interaction among limbs. For instance, Fig. 1 illustrates the balance beam event in gymnastics, where fine-grained actions such as ' salto backward tucked ' and ' salto backward tucked with one twist '

See https: // www.ieee.org / publications / rights / index.html for more information.

Fig. 1. Examples of ' salto backward tucked ' and its variant with one twist reveal the fine-grained nature of action recognition, characterized by large intraclass variation and subtle interclass di GLYPH&lt;11&gt; erences. Accurate discrimination relies more on capturing the temporal dynamics of local movements than on appearance or context. (a) Salto backward tucked in the balance beam. (b) Salto backward tucked with one twist in the balance beam.

<!-- image -->

Fig. 2. Overall framework of our proposed ART. The backbone extracts the feature from an input video, the spatial semantic extraction (SSE) component enhances the feature with the spatial context, the RSSA component captures region-specific semantic responses from enhanced region-wise representations, and the TG component forms a group of action tracklets, i.e., a group of responses to the same position queries from all video frames along the temporal dimension. Finally, tracklet-based representations are integrated into a global representation through tracklet aggregation (TA), obtaining the video's final recognition result. Furthermore, we transform action label descriptions into action phrases aligned with VLMs' textual lexicon, forming a text-constrained semantics bank.

<!-- image -->

are subtly di GLYPH&lt;11&gt; erentiated by the gymnast's execution of a twist during the backward flip. Consequently, the recognition models should have a robust capability for discovering and representing distinct local details and their dynamics, which is essential for achieving accurate fine-grained action understanding.

Substantial e GLYPH&lt;11&gt; orts have been dedicated to the exploration of crucial region localization and interaction in vision recognition tasks [21], [22], [23], [24], [25], [26]. For instance, Fayyaz et al. [21] developed an adaptive token sampler for vision Transformers (ViTs) that scores and selectively samples crucial tokens, improving e GLYPH&lt;14&gt; ciency without compromising performance. STMixer [27] introduced a query-based adaptive feature sampling module to flexibly extract discriminative spatiotemporal features, yielding an e GLYPH&lt;14&gt; cient end-to-end action detector. TempMe [28] proposed a parameter-e GLYPH&lt;14&gt; cient and computation-e GLYPH&lt;14&gt; cient framework for text-video retrieval by addressing temporal redundancy across video frames. TempMe leveraged a progressive multigranularity temporal token merging strategy to significantly reduce trainable parameters, token count, and model complexity. ALIN [25] dynamically identified the most informative tokens at a coarse granularity and subsequently refined these located tokens to a finer granularity, enabling the exploration of valuable fine-grained spatiotemporal interactions. These approaches aim to diminish the number of tokens processed during various feature extraction phases, thereby significantly reducing computational costs. However, this selective filtering process often results in a set of discontinuous tokens across both space and time, disrupting the continuity of action regions. Specifically, if tokens identified as crucial in one frame are missing or only partially captured in subsequent frames, it complicates the accurate modeling of action sequences, leading to potential inaccuracies in recognition tasks. A straightforward solution involves explicitly locating and tracking discriminative, actionspecific local regions as an auxiliary task. However, the resource-intensive nature of video data makes training endto-end models that manage both localization and tracking particularly challenging, especially in the absence of detailed pixel-level or region-level annotations.

Fig. 3. (a) Visualization illustrates the motivation behind our ART framework. Our ART aims to identify and track discriminative action regions across multiple local areas that evolve over time. Here, X represents semantic features extracted from the backbone network, while Tr denotes features processed by ART. (b) and (c) CAMs and the response regions contributing to action prediction without and with ART, respectively. We can see that the backbone network tends to concentrate on easily distinguishable regions, often overlooking the dynamics of local details. In contrast, our ART framework focuses on discriminative action regions over time.

<!-- image -->

These limitations motivate us to design a streamlined and e GLYPH&lt;11&gt; ective framework, which implicitly identifies and tracks distinctive local details along the temporal dimension in a self-supervised manner, where only video-level labels are available. Building on the success of query-based mechanisms in Transformers for object detection (e.g., EDTR [29] and UPDETR [30]) and instance segmentation (e.g., VisTR [31] and E GLYPH&lt;14&gt; cientVIS [32]), we adapt this approach to track actionspecific regions in video sequences. Rather than relying on implicit spatiotemporal encoding, explicitly discovering and organizing local action regions into tracklets facilitates better reasoning over motion dynamics. These tracklets provide localized, structured motion representations that not only enhance recognition accuracy but also improve the interpretability of model predictions. To this end, we propose a novel action-region tracking (ART) framework, as shown in Fig. 2. Initially, ART employs a standard video encoder to sequentially extract region-level features from the input video. Second, ART introduces a region-specific semantic activation module that employs discriminative and text-constrained semantics serve as queries to capture the most action-related region responses in each video frame, facilitating interaction among spatial and temporal dimensions with corresponding video features. The text-constrained queries are designed to expressly encode nuanced semantic representations derived from the textual descriptions of action labels, as extracted by the language branches within visual language models [33]. Notably, these semantic representations are stored in memory, allowing for e GLYPH&lt;14&gt; cient retrieval without real-time processing by vision language models (VLMs) during inference. Subsequently, the region-specific semantic responses are organized into a group of action tracklets, in which each tracklet links related responses over time, facilitating detailed regional action tracking. In Fig. 3(c), we illustrate how individual region responses are grouped into action-region tracklets that follow consistent spatial-temporal dynamics. These tracklets are aligned to meaningful subactions (e.g., takeo GLYPH&lt;11&gt; and mid-air twist), o GLYPH&lt;11&gt; ering an intuitive depiction of how ART disentangles motion cues. In contrast, the class activation maps (CAMs) [as shown in Fig. 3(b)] of the backbone network tend to highlight broad and coarse regions, such as the entire body, without exhibiting spatial selectivity toward meaningful subparts. This process disentangles the overall spatiotemporal representation into distinct tracklets corresponding to di GLYPH&lt;11&gt; erent body parts, thus sharpening the semantic distinctions and enhancing the characterization of fine-grained action di GLYPH&lt;11&gt; erences.

However, in the absence of region-level annotations, accurately constraining these region queries to capture specific action details becomes di GLYPH&lt;14&gt; cult. To address this, we introduce a multilevel tracklet contrastive loss (MTC-Loss) that operates on region-aware semantic responses at spatial, temporal, and tracklet levels. It captures diverse semantic responses in each frame and models high correlations among similar response regions across adjacent frames. This loss function serves as a robust self-supervised constraint that: 1) distinguishes individual region responses within each frame at the spatial level; 2) establishes correlations among region responses across adjacent frames at the temporal level; and 3) promotes the generation of diverse tracklets at the tracklet level.

Given the minimal variations among category names and the fact that the quality of action-region localization is determined by their semantic representations, this situation can result in a text semantic space with insu GLYPH&lt;14&gt; cient discriminative power. This lack of specificity may introduce ambiguity in the final recognition process. To mitigate this issue, we consider the adaptability of semantic representations to each individual video instance, tailoring them more closely to the unique characteristics of each video. Initially, we introduce a set of learnable prompts designed to integrate semantic representations into the region-specific semantic activation module. This integration allows for the nuanced capture of individual variations within each video. Furthermore, we implement a task-specific fine-tuning mechanism inspired by the exponential moving average (EMA) approach. This mechanism refines

the textual semantics during training, preserving the core semantic representations encoded by VLMs while optimizing them for task-specific preferences.

The proposed ART demonstrates robust performance in FGAR, achieving 91.2% Top-1 accuracy on FineGym99 and 84.4% Top-1 accuracy on FineGym288. These results significantly surpass the current state of the art. This represents a clear advancement over the previous state-of-the-art model, MDCN [34], improving by margins of + 1.3% and + 1.0% on FineGym99 and FineGym288, respectively. In addition, ART achieves these results with reduced computational complexity compared with full Transformer-based methods such as ViViT [35] and MViT [36]. The key contributions are summarized as follows.

- 1) We propose a novel ART framework to discover and track distinctive details of local regions in a video frame to enrich the encoding of spatial and temporal contexts for better reasoning fine-grained actions.
- 2) We devise an MTC-Loss to guide ART to accurately capture action details in an e GLYPH&lt;11&gt; ective self-supervised manner at spatial, temporal, and tracklet levels.
- 3) We introduce a task-specific fine-tuning mechanism designed to enhance textual semantics, which retains the semantic representations encoded by VLMs and optimizes them to align with the specific requirements of the downstream task.
- 4) The results of extensive experiments conducted on four widely used fine-grained and conventional action recognition datasets, including FineGym [14], Diving48 [37], NTURGB-D [38], and Kinetics [11], clearly demonstrate the superiority of our proposed method against the state of the art.

## II. RELATED WORKS

## A. Action Recognition

Recognizing human behaviors is a fundamental task in computer vision, involving the classification of observed actions based on video input. This process is pivotal for applications ranging from surveillance to interactive systems. Typically, there are four types of deep-based models: two-stream model, 3-D model, (2 + 1)D model, Transformer-based model.

- 1) Two-Stream Model: The two-stream architectures [39], [40], [41] process RGB frames and optical flow images through separate CNNs to extract features, which are then combined. This model excels in capturing motion but may lack temporal context in its analysis.
- 2) 3-D Model: Since videos can be viewed as temporally dense sequences of image samples, extending 2-D convolution operations of 2-D CNN to 3-D convolutions is the most intuitive method for spatiotemporal feature learning. For example, various 3-D convolutional networks [11], [18], [42], [43] have been proposed to directly learn spatiotemporal features from RGB frames. C3D [42] was the first work utilizing deep 3-D CNNs for learning spatiotemporal features. I3D [11] introduced a new two-stream inflated 3-D network that models spatial and motion features with a visual stream and a flow stream. SlowFast [18] involved a slow pathway and a fast

pathway to capture spatial semantics at a low frame rate and motion information at a fast frame rate, respectively.

3) ( 2 + 1 )D Model: 3-D networks generally su GLYPH&lt;11&gt; er from heavy computational cost. To reduce computational cost, various methods [5], [6], [15], [44], [45] were proposed to decompose 3-D convolutions into 2-D spatial and 1-D temporal filters. STM [6] learns spatiotemporal and motion features from shared feature maps, while TEA [45] leverages temporal di GLYPH&lt;11&gt; erences with a hierarchical residual design to capture both short- and long-range motions.

- 4) Transformer-Based Model: Inspired by the significant advancements achieved by Transformers in natural language processing, the ViT [46] was naturally developed from image recognition tasks to video recognition tasks [9], [17], [20], [35], [47], [48], [49]. ViViT [35] extended pure Transformers to video classification. TimeSformer [47] tailored self-attention for spatiotemporal learning. MViT [20] adopted multiview encoders with lateral connections to fuse temporal information.

Despite progress, existing models struggle with videos di GLYPH&lt;11&gt; ering only in subtle motions and temporal dynamics, often overlooking local nuances. To address this, our work enhances recognition of fine-grained motion and temporal details, advancing action recognition performance.

## B. Fine-Grained Action Recognition

FGAR, which focuses on distinguishing subtle interclass di GLYPH&lt;11&gt; erences among closely related action categories, has garnered increasing attention due to its complexity and applicability. Compared with general action recognition, FGAR is distinguished by several critical characteristics: 1) significant intra-class variation coupled with subtle interclass di GLYPH&lt;11&gt; erences; 2) the insu GLYPH&lt;14&gt; ciency of sparse sampling frames for representing actions; 3) the paramount importance of motion information over visual appearance; and 4) the necessity for precise temporal dynamics modeling. As a foundation for more complex technologies, the pursuit of better datasets in the field of action understanding has never ceased. Early, several datasets have been developed to facilitate research in this area, such as Diving48 [37], Something-Something [13], and NTURGB-D [38], each designed to explore nuanced human-object interactions. For instance, Diving48 [37] focused on the recognition and analysis of complex diving actions, classifying performances based on various actions and orientations like somersaults , twists , and handstands . Something-Something [13] encompassed 147 classes of routine human-object interactions, capturing diverse activities ranging from moving objects down to retrieving items from specific locations. Most recently, FineGym [14], crafted from high-definition gymnasium videos, included intricate motion details exemplified by actions like vault-women: double salto backward tucked . FineGym not only provided a new and large-scale benchmark but also had verified that current models were still inadequate in capturing the nuanced spatiotemporal semantics required for fine-grained recognition. The focus of research is toward the development of e GLYPH&lt;11&gt; ective approaches for fine-grained video understanding. TQN [10] leverages multiattribute sublabels from action texts for granular attribute learning with Transformers. MDCN [34]

disentangles motion features to emphasize dominant dynamics. CANet [50] employs class-specific attention and dictionary learning to decouple class-wise features. PGVT [51] integrates pose priors and temporal attention for enhanced FGAR.

Despite these advancements, existing models have not fully addressed the need for discriminative local motion representations that consider the dynamics of local action details. In this work, we propose to discover and track di GLYPH&lt;11&gt; erent action regions along the temporal dimension for more accurately characterizing fine-grained actions.

## C. Discriminative Region Localization

The human brain employs a hierarchical and diverse scale of attention to process visual information, boosting its capacity to discern vital information from the environment while selectively ignoring the insignificant details. Leveraging the characteristics of serialized tokens in ViTs and their capacity to capture long-range temporal dependencies, significant research e GLYPH&lt;11&gt; orts have been focused on pinpointing and examining crucial regions (tokens) and their interactions in the realm of image and video recognition tasks [21], [22], [23], [25], [52], [53], [54], [55]. The patterns of discriminative region localization can broadly be categorized into three types.

- 1) Attention-Based [56], [57], [58], [59]: Motionformer [56] employed trajectory attention to model long-range temporal dependencies via continuous token paths, while Korban et al. [58] proposed a spatiotemporal Transformer with semantic attention and motion-aware encoding to capture spatial-motion interactions and dynamic variations.
- 2) Token Sampling [21], [22], [25], [60]: Retaining partially class-discriminative tokens within the sequence of tokens. DynamicViT [22] introduced a framework for dynamic token sparsification, aimed at progressively pruning redundant tokens. C2F-ALIN [25] dynamically identified the most informative tokens with coarse granularity and subsequently divided these tokens into finer granularity to facilitate detailed spatiotemporal interaction.
- 3) Token Fusion [23], [52], [55]: Aggregating token representations based on their semantics through the depth of a visual Transformer. EViT [55] enhances token selection by retaining attentive tokens and fusing inattentive ones via gradient back-propagation, while ToMe [61] accelerates Transformers by merging similar tokens through a lightweight matching algorithm, combining pruning speed with improved accuracy.

In addition to the works that aim to accelerate the inference of convolutional neural networks, other works aim to improve the e GLYPH&lt;14&gt; ciency of Transformer-based models.

## D. Contrastive Learning

Contrastive learning [62], [63], [64], [65], [66], [67], [68], [69], [70] has demonstrated its great potential to learn discriminative features in a self-supervised manner. The key idea of contrastive learning lies in maximizing the similarity of representations among positive samples while exploiting discriminative patterns between positive and negative samples. Contrastive learning can be e GLYPH&lt;11&gt; ectively implemented by appropriately defining positives and negatives in relation to the video recognition tasks, which include augmented transformations [71], multiview perspectives [72], and temporal coherence [73]. Our work departs from image-level features by learning region-based discriminative motion representations via contrastive learning and further leveraging them to track action-specific details that enhance FGAR.

## III. METHOD

The overview of our proposed ART is shown in Fig. 2, consisting of five main components: 1) spatiotemporal feature extraction; 2) region-specific semantics activation; 3) tracklet generation (TG); 4) TA; and 5) text-constrained semantics bank. First, we extract the spatiotemporal feature with a video encoder. Second, we introduce a set of text-constrained semantics serving as region queries to focus on regions where a specific atomic activity occurs. Third, the identified regions of all frames are organized into action tracklets, which characterize the region-based action dynamics by linking related responses across di GLYPH&lt;11&gt; erent video frames, even without the need for supervision signals such as object proposals. Meanwhile, we design a multilevel tracklet contrastive constraint to optimize generated action tracklets. Fourth, all action tracklets are aggregated to learn the dynamic representation of distinctive local details, which are then used for the final action prediction. Finally, in training, we refine the text-constrained semantics bank to ensure the semantic representations encoded by VLMs are not only preserved but also optimized for specific task preferences.

## A. Spatiotemporal Feature Extraction

In the feature extraction stage, we sample T frames of a given video as input V 2 R T × H 0 × W 0 × 3 , where H 0 × W 0 is the spatial resolution of input frames. Following the origi nal UniFormer [17], we initially apply 3-D convolution to project the input video into spatiotemporal tokens. We then employ Transformer layers for feature extraction, resulting in 1 + ( T × H × W ) tokens with the feature channel dimension C , which includes one class token and T × H × W visual tokens (where H × W represents the spatial resolution of the features, which is 1 / 16 the size of the original dimensions H 0 × W 0). The obtained spatiotemporal feature is denoted as X 2 R T × H × W × C .

## B. Region-Specific Semantics Activation

In order to capture the region-specific semantic responses from the visual tokens of an individual frame, we propose a region-specific semantics activation (RSSA) module in this work. This module comprises a spatial semantics enhancement stage and an RSSA stage.

- 1) Spatial Semantic Enhancement Stage: First, to enhance the spatial semantics and mitigate the impact of noise in the background, we introduce the TopK action-related textconstrained semantics S topk 2 R K × C from the textural semantics bank (we will provide a detailed explanation of the construction and updating of the textural semantics bank in

Section III-E). Where, text-constrained semantics S topk possess the capability to focus on the action regions. Although the textual queries are static and defined at the dataset level, their corresponding visual responses are learned dynamically, conditioned on the specific content of each video. This design allows for generalizable, semantically grounded localization without the need for video-specific text generation. As illustrated in Fig. 2, the spatial features of each frame ( X t 2 R HW × C ; t = 1 ; 2 ; : : : ; T ) is enhanced as follows:

<!-- formula-not-decoded -->

where Concat( GLYPH&lt;1&gt; ) denotes the concatenate operation and MSA( GLYPH&lt;1&gt; ) denotes the spatial semantic enhancement function, which is employed as multiheaded self-attention. Then, we obtain the processed spatial feature ˆ X t that focuses more on regions relevant to the action semantics. Similarly, the text-constrained semantics ˆ S topk t are fine-tuned according to the specifics of the instance.

2) RSSA Stage: As illustrated in Fig. 2, to obtain actionrelated region responses that are only activated for the regions within the frame, the enhanced spatial feature of each frame interacts with the region queries. Inspired by DETR, we incorporate a new component which introduces a set of learnable prompts P t 2 R K × C designed to integrate semantic representations, the text-constrained semantics ˆ S topk t serving as region queries

<!-- formula-not-decoded -->

This integration allows the region queries Q t to not only perceive action-related semantics but also capture individual variations within each video. We then employ the crossattention mechanism to perform the RSSA. Specifically, given a spatial feature ˆ X t 2 R WH × C and a set of region queries Q t 2 R K × C , the region-specific semantic responses are obtained through a multiheaded cross-attention operation

<!-- formula-not-decoded -->

where R t = f r t ; 1 ; r t ; 2 ; : : : ; r t ; K g 2 R K × C , MCA( GLYPH&lt;1&gt; ) denotes the multiheaded cross-attention function, the Softmax( GLYPH&lt;1&gt; ) function is performed along WH dimension, and W Q , W K , and W V are linear projection layers to project Q t and ˆ X t into the latent space.

Correspondingly, the set of region-specific semantic responses for all frames f R 1 ; R 2 ; : : : ; R T g is obtained in parallel by the T RSSA modules.

## C. Tracklet Generation

- 1) Tracklet Generation: After obtaining the regionspecific semantic responses from all frames, denoted as R 1 ; R 2 ; : : : ; R T , we construct temporal action tracklets by temporally aligning responses corresponding to the same semantic query across frames. Each R t = r t ; 1 ; r t ; 2 ; : : : ; r t ; K contains K region-specific embeddings extracted at frame t via cross attention with the set of query vectors Q t . We assume that each query index k corresponds to a consistent semantic concept

across all frames (e.g., 'arm extension' or 'leg twist'). Thus, we naturally define the k th action tracklet Tr k by temporally concatenating the k th region response from every frame

<!-- formula-not-decoded -->

Each tracklet Tr k can be interpreted as the temporal evolution of a semantically meaningful local region, governed by a fixed text-constrained query. This design provides two major advantages: 1) it avoids the need for external supervision or heuristic region association (e.g., optical flow) and 2) it enables structured temporal modeling at the semantic region level. The semantic identity of each tracklet is preserved across time by consistent indexing, while the region responses r t ; k are adaptively updated by content-specific video features during training. The resulting tracklets serve as inputs to the subsequent MTC-Loss, which refines their discriminability and temporal alignment via spatial, temporal, and tracklet-level constraints.

- 2) Multilevel Tracklet Contrastive Loss: Our framework obtains a fixed-size sequence of K region responses for each frame through the RSSA module. Once the tracklets are obtained, the main challenge is to constrain these tracklets to capture accurate action details without the supervision of region-level annotations. Thus, to capture diverse semantic responses in each frame and model high correlations of similar response regions between adjacent frames, we introduce an MTC-Loss among region-aware semantic responses at spatial, temporal, and tracklet levels, which e GLYPH&lt;11&gt; ectively distinguishes individual region responses of each frame (spatial level), establishes correlative region responses between adjacent frames (temporal level), and generates diverse tracklets (tracklet level) in a self-supervised manner.
- a) Spatial-level tracklet contrastive loss: Active region responses extracted from a frame should locate the discriminant di GLYPH&lt;11&gt; erences so as to capture rich and discriminative fine-grained features. We repel the region responses in the same frame with spatial-level tracklet contrastive loss

<!-- formula-not-decoded -->

where &lt; GLYPH&lt;1&gt; &gt; is computed based on the cosine similarity.

- b) Temporal-level tracklet contrastive loss: Active region responses extracted from each frame of a video are not independent but correlated to an ongoing action instance collectively. We attract the responses from di GLYPH&lt;11&gt; erent frames individually, located in the same order of each frame. So the spatial-level tracklet contrastive loss is defined as follows:

<!-- formula-not-decoded -->

where GLYPH&lt;21&gt; is a hyperparameter denoting the correlated degree of the responses from adjacent frames.

- c) Tracklet-level tracklet contrastive loss: Furthermore, the tracklet representations should be di GLYPH&lt;11&gt; erent from each other

so as to track the diverse details in a video. Our tracklet-level tracklet contrastive loss is defined as follows:

<!-- formula-not-decoded -->

Our MTC-Loss is formed as follows:

<!-- formula-not-decoded -->

## D. Tracklet Aggregation and Prediction

- 1) Tracklet Aggregation: Mean pooling is a commonly employed method for aggregating the semantic responses of each tracklet to generate the final tracklet representation. As illustrated in Fig. 2, instead of treating each semantic response equally as in mean pooling, we propose a weighted aggregation that utilizes the action-related text-constrained semantics S topk to capture the temporal saliency for TA. For a set of responses Trk = f r 1 ; k ; r 2 ; k ; : : : ; r T ; k g of k th tracklet and a set of action-related text-constrained semantics S topk = f s 1 ; s 2 ; : : : ; s K g , we compute the similarity between each response and each semantic embedding to assess the fine-grained relevancy. Next, we apply a softmax operation to normalize the similarities for each tracklet, and subsequently aggregate the similarities between a given tracklet and various semantics to derive a tracklet-level saliency

<!-- formula-not-decoded -->

where Tr agg k 2 R 1 × C is the final aggregated tracklet representation.

2) Prediction: A supervised task is introduced to train the backbone network for extracting task-relevant spatiotemporal feature X and modeling global action representation x cls (i.e., class token)

<!-- formula-not-decoded -->

where y is the action ground truth, p ( GLYPH&lt;1&gt; ) is the predicted probability of a whole input video, and B is the number of training samples.

Finally, these tracklet representations are integrated with the global representation of the whole video for producing final recognition results

<!-- formula-not-decoded -->

## E. Text-Constrained Semantics Bank Generation and Fine-Tuning

- 1) Text-Constrained Semantic Bank Generation: We first convert category descriptions of action labels into action phrases that approximate the pretrained textual lexicon in VLMs. For example, using phrases like 'This is a video about f action label g ' or 'The person is doing f action label g ,' we

Fig. 4. Illustration of the process of task-specific textual semantic bank finetuning. We implement a task-specific fine-tuning mechanism for updating S , ensuring that the textual semantics retain the semantic representations encoded by VLMs while fine-tuning to align with task-specific preferences. The agent textual semantic bank S a is optimized by a video consistency loss and a prototype consistency loss to narrowed the distance between categories across video and text modalities.

<!-- image -->

generate N prom prompt templates for each action category. Next, we utilize the pretrained text encoder branch of CLIP [33] to extract semantic features from these prompt templates, initializing the textual semantic bank S 0 2 R N prom × N class × Ct , where N class is the number of classes to classify, and Ct is the dimension of textual semantics. S 0 is then updated through an EMA-like process, fine-tuning the text-constrained semantic bank to produce a highly distinct semantic bank S 2 R N prom × N class × Ct .

- 2) Text-Constrained Semantic Bank Fine-Tuning: We implement a task-specific fine-tuning mechanism for updating S 0 , ensuring that the textual semantics retain the semantic representations encoded by VLMs while fine-tuning to align with task-specific preferences. As shown in Fig. 4, after extracting the initial textual semantic bank S 0 , we duplicate it as an agent textual semantic bank S a and then truncate the gradients of S . The task-specific textual semantic fine-tuning is formulated as follows:

<!-- formula-not-decoded -->

where and

L

GLYPH&lt;17&gt;

is the learning rate, sema

GLYPH&lt;22&gt;

is the momentum update rate,

S

[in (17)] is the optimization function for training

a

t

.

Agent Textual Semantic Bank Training:

- 1) Video Consistency Loss: In the textual semantic bank, each class is represented by a total of N prom semantics according to N prom prompt templates. Therefore, video consistency loss L video consist is used to ensure the cosine consistency between the video representation x cls and the textual semantics corresponding to the respective category. Specifically, alignment is implemented through winner-take-all classification

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S cls 2 R 1 × N class is a classification score.

- 2) Prototype Consistency Loss: Prototype consistency loss L prot consist is devised to ensure the cosine consistency between the action prototypes W and the textual semantics corresponding to the respective category, which can be understood as the center cluster of each class, i.e., class prototype. The class prototypes W 2 R N class × D are decoupled from the prediction head. L prot consist is defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where S prot 2 R N class × N class is a similarity matrix. Therefore, the ultimate objective function for training the agent textual semantic bank is denoted as follows:

<!-- formula-not-decoded -->

Our ART is trained using the total loss function

<!-- formula-not-decoded -->

where L v and L con are the supervised learning losses for the global representation-based prediction and the final prediction, respectively, L MTC denotes the MTCLoss. L sema denotes the text-constrained semantic bank fine-tuning loss. We empirically set the hyperparameters GLYPH&lt;13&gt; 1 = GLYPH&lt;13&gt; 2 = 5 in our experiments so that none of the loss terms dominates the training. During the inference, action predictions are generated by the final prediction [in (11)].

## IV. EXPERIMENTS AND DISCUSSION

## A. Implementation Details

We use CNN-based TEA [45] and Transformer-based UniFormerV2 [49] as the backbones for feature extraction, respectively. We evaluate the performance of ART on four widely used action recognition datasets. FineGym [14], released very recently, is a large-scale, high-quality action dataset with fine-grained annotations, which was collected from HD gymnasium videos of 288 categories. Diving48 [37] contains 48 categories with 16k training and 2k testing of competitive diving videos. NTURGB-D [38] is a large-scale RGB-D dataset with synchronized skeleton and video data, widely used for action recognition and pose estimation, and includes NTU-60 and NTU-120 subsets. Kinetics-400 / -600 [11] is a widely used large-scale benchmark dataset for action recognition, consisting of 55 h of recordings capturing daily activities. The training parameters for all the datasets are: 50 training epochs, an initial learning rate of 0.01 (decreased by 0.1 for every 20 epochs), and a dropout rate of 0.5. The sparse sampling strategy [11] is used to extract T frames from videos ( T = 16). Each input frame is cropped and resized to 224 × 224 for training and testing. Random scaling, cropping, and horizontal flipping are deployed as data augmentation.

We use a standard Transformer layer [46] with four layers in spatial enhancement (SE) and RSSA, respectively. The dimension of the latent space d is 256, and the number of queries of each frame K (also known as the number of tracklets) is set as 2. The chosen hyperparameters in our approach are GLYPH&lt;13&gt; 1 = GLYPH&lt;13&gt; 2 = 5 and GLYPH&lt;21&gt; = 0 : 6. The ablation studies presented below will demonstrate the e GLYPH&lt;11&gt; ectiveness of our configurations.

## B. Comparison With the State of the Art

As our work is focused on FGAR, the evaluation is mainly conducted on three publicly available fine-grained action datasets, FineGym [14], Diving48 [37], and NTURGBD [38]. As shown in Table I, we provide the overall performance comparison between our proposed ART and other state-of-the-art methods in terms of two metrics, perclass accuracy (Mean%) and per-video accuracy (Top-1%). Quantitative results are listed in Table I, where models are grouped into two categories: CNN-based and Transformerbased. From these results of CNN-based models, we can see that the overall performance of ART outperforms the most competitive existing method, MDCN, on all three datasets. From these results of Transformer-based models, our ART achieves 94.7%, 90.2%, and 87.9 Top-1 accuracies, respectively, improving UniFormerV2 by 1.8%, 3.0%, and 2.0% on all three datasets. Our ART only falls behind TQN in the Mean accuracy on the FineGym288 dataset. TQN was introduced using a Transformer architecture to facilitate the learning of granularity across di GLYPH&lt;11&gt; erent attributes, guided by the supervision of multiattribute sublabels derived from multipart text descriptions of action labels. TQN also used more input frames (i.e., 48 frames over the popular setting of 16 frames) to achieve a Mean accuracy of 61.9%. Our ART achieves competitive results with fewer input frames. These experimental results clearly demonstrate the e GLYPH&lt;11&gt; ectiveness of our action tracklet network for FGAR. We compare our ART with other video + text baselines in Table I. Notably, although ActionCLIP and X-CLIP, and Wu et al. [79] utilize d similar external textual supervision, they still fall behind our ART model in all metrics across datasets. For instance, on FineGym99, our ViTL / 14-based ART achieves 90.9% Top-1 accuracy, significantly outperforming X-CLIP (89.6%) and Wu et al. [79] (88.9%). Similar trends are observed on Diving48 and FineGym288. These results demonstrate that the performance gain is not simply due to access to text data, but stems from our taskaligned tracklet-based architecture and the MTC-Loss, which better leverage textual semantics in a temporally consistent and fine-grained manner. In addition to the accuracy metrics, we also report inference cost in terms of FLOPs. It is noticed that our ART has similar computational costs due to the adoption of the same backbones. Since we use a lightweight Transformer architecture, ART's FLOPs are still at a low level.

We further assessed the performance of our ART on the NTU-60 and NTU-120 datasets, analyzing the cross-subject (XSub), cross-view (XView), and cross-setup (XSet) scenarios individually for each dataset. Table II showcases the recognition performance comparison across di GLYPH&lt;11&gt; erent video

TABLE I COMPARISON OF RECOGNITION PERFORMANCE ON FINEGYM99, FINEGYM288, AND DIVING48. AVERAGED PER-CLASS ACCURACY (MEAN%), TOP-1 ACCURACY (%), AND FLOPS ARE SHOWN. RED/BLUE INDICATE SOTA/THE SECOND BEST

TABLE II COMPARISON OF RECOGNITION ACCURACY ON THE NTURGB-D WITH CROSS-SUBJECT (XSUB), CROSS-VIEW (XVIEW), AND CROSS-SETUP (XSET) SETTINGS IN TERMS OF TOP-1 ACCURACY (%). RED/BLUE INDICATE SOTA/THE SECOND BEST

classification benchmarks. The results underscore the consistent superiority of ART over established baselines.

We also evaluate the generalization potential of our model for conventional action recognition on Kinetics-400 and Kinetics-600 that are less sensitive to fine granularity and the dynamics of distinctive local details. As shown in Table III, our ART outperforms other existing methods by achieving 90.3% and 89.9% Top-1 accuracy on Kinetics-400 and Kinetics-600 datasets. These results clearly demonstrate the e GLYPH&lt;11&gt; ectiveness

TABLE III RECOGNITION PERFORMANCE ON KINETICS-400 AND KINETICS-600 IN TERMS OF TOP-1 ACCURACY (%) AND TOP-5 ACCURACY (%). RED/BLUE INDICATE SOTA/THE SECOND BEST

and generalization of our model for conventional action recognition.

## C. Ablation Study

We conduct extensive ablation studies to evaluate the impact of various design choices, including the SE stage, RSSA module, and TA module. Ablation studies are conducted on FineGym99, Diving48, and NTU60-XView, using UniFormerV2 (ViT-B / 16) as the backbone. As shown in Table IV, the experiments are conducted under five settings.

TABLE IV IMPACT OF THE KEY COMPONENTS IN ART ON FINEGYM99, DIVING48, AND NTU60-XVIEW IN TERMS OF TOP-1 AND MEAN ACCURACIES (%)

- 1) Baseline: The baseline starts from the backbone network by simply feeding the backbone's output features of a given video into a classifier for action prediction;
- 2) Baseline + RSSA: The output features of the backbone are fed into the RSSA module to obtain region-specific semantic responses, which are then arranged into tracklets. In this case, we used a simple strategy of directly adding the elements to aggregate the tracklets. The aggregated tracklet representation is directly concatenated with the video class token x cls for final prediction;
- 3) Baseline + RSSA (w / o SE): To verify the e GLYPH&lt;11&gt; ectiveness of the SE stage in the RSSA module, we conducted experiments by removing the SE.
- 4) Baseline + RSSA + TA: Unlike Setting (2), we performed TA module to merge all the tracklets.
- 5) Baseline + RSSA + TA w / MTC-Loss: For generated tracklets, we introduce MTC-Loss to constrain these tracklets to capture action details accurately.
- 1) E GLYPH&lt;11&gt; ectiveness of SE: From Table IV, we can observe significant performance improvement of Setting (2) over the baseline Setting (3), improved by 0.5%, 0.8%, and 0.9 Top-1 accuracies on FineGym99, Diving48, and NTU60-XView, respectively. This clearly shows that our SE, i.e., the encoder architecture, is able to capture saliency semantics in the spatial dimension and is more e GLYPH&lt;11&gt; ective for FGAR, compared with the backbone.
- 2) E GLYPH&lt;11&gt; ectiveness of Action Tracklet (RSSA and TA): As described in Sections III-B and III-D1, we obtain regionspecific semantic responses through RSSA and aggregate action tracklets through TA for the final prediction. To demonstrate the e GLYPH&lt;11&gt; ectiveness of the action tracklet, we implement our model under Settings (2) and (4) in Table IV, where the only di GLYPH&lt;11&gt; erence is the usage of our proposed TA module. It can be observed that Setting (4) gains a significant performance improvement compared with the baseline. This fully verifies the importance of considering region-based action dynamics for action recognition. Moreover, Setting (4) also achieves a performance improvement over Setting (2). This demonstrates that simply generating action tracklets may not capture the discriminative ones. After introducing the TA module, better performance is achieved.

We also evaluate the impact of the number of action tracklets under Setting (4) in Fig. 5(a). It is also equivalent to the number of the discriminative regions of interest in a frame. As the objects of interest and background in fine-grained action videos are generally the same or very similar, discriminative

Fig. 5. Impact of (a) correlation degree of GLYPH&lt;21&gt; and (b) number of action tracklets on FineGym99, Diving48, and NTU60-XView in terms of Top-1 accuracy.

<!-- image -->

action-specific di GLYPH&lt;11&gt; erences only occur in several regions. We investigate the e GLYPH&lt;11&gt; ect of varying the number of action tracklets K , and find that the best performance is achieved when K = 2.

- 3) E GLYPH&lt;11&gt; ectiveness of MTC-Loss: We investigate the performance of our ART with di GLYPH&lt;11&gt; erent components of MTC-Loss. As shown in Table V, the final case with all three loss terms consistently outperforms the other cases. Starting from the baseline (the first case), adding L spatial or L tracklet alone can improve the performance, respectively. When they are integrated together (the fifth case), the performance increases further. We should note that using L temporal alone may decrease the performance (even lower than that of the baseline); however, introducing it together with both spatial and tracklet losses (the last case) can boost the performance further. It demonstrates that only when the meaningful responses are discovered at intra-frame level, L temporal can work well to link these responses together as the tracklets. To conclude, MTC-Loss can help capture action details in each frame accurately and track the active regions over time, thereby further enhancing discriminative representation for FGAR.

In Fig. 5(b), we evaluate the impact of the correlation degree GLYPH&lt;21&gt; of L temporal . GLYPH&lt;21&gt; denotes the correlation degree between neighboring frames, and it does a GLYPH&lt;11&gt; ect the quality of action tracklets. Due to limb deformation and view change for adjacent frames, GLYPH&lt;21&gt; is set to 0.6 to obtain the best results supported by experimental validation.

- 4) E GLYPH&lt;11&gt; ects of the Task-Specific Textual Semantic FineTuning: In our investigation, we focused on the fine-tuning mechanism of the textual semantic bank S , specifically employing an EMA approach supervised by cosine similarity loss to advance textual semantics for cross-modality alignment. As a result, substantial performance improvements were observed on the FineGym99, Diving48, and NTU60-View

TABLE V

IMPACT OF EACH COMPONENT IN MTC-LOSS ON FINEGYM99, DIVING48, AND NTU60-XVIEW IN TERMS OF TOP-1 AND MEAN ACCURACIES (%)

TABLE VI IMPACT OF EACH COMPONENT IN TASK-SPECIFIC TEXT-CONSTRAINED SEMANTIC BANK FINE-TUNING ON FINEGYM99, DIVING48, AND NTU60-XVIEW IN TERMS OF TOP-1 AND MEAN ACCURACIES (%)

datasets compared with the baseline. In Table VI, we further validate the e GLYPH&lt;11&gt; ectiveness of our proposed video consistency loss L video consist and prototype consistency loss L prot consist . Under the supervision of L video consist and L prot consist , the textual semantic bank updated strategy not only established stronger consistency between di GLYPH&lt;11&gt; erent textual semantics of the same category but also narrowed the distance between categories across video and text modalities. Comparative results indicate that the implementation of consistency losses led to a more notable enhancement in performance. As shown in Fig. 6, to illustrate that the task-specific textual semantic fine-tuning mechanism e GLYPH&lt;11&gt; ectively updates the initial textual semantic, we first visualize the semantic embeddings of the textual semantic bank using t-SNE without (left) and with (right) task-specific finetuning. The embeddings are more dispersed compared with those of the baseline, which indicates that our method has the ability to extract more discriminative category semantics. This significantly enlarges the interclass distribution di GLYPH&lt;11&gt; erences, thereby improving the performance of FGAR.

## D. Complexity Analysis

To further assess the e GLYPH&lt;14&gt; ciency-performance tradeo GLYPH&lt;11&gt; of the proposed ART framework, we present a comparison of computational complexity, model size, and recognition accuracy against a strong baseline, UniFormerV2, as summarized in Table VII. The ART model incurs a modest increase in GFLOPs, rising from 1332.43 to 1429.32, representing a 7.27% increase in computational cost. Similarly, the parameter count grows from 345.05 million to 365.01 million, an increase of 5.78%. Despite these slight increases in resource demand, ART achieves a substantial performance gain, improving the Top-1 accuracy from 90.5% to 93.5% ( + 3.0%) and the Mean accuracy from 85.7% to 89.2% ( + 3.5%). We conduct a training

Fig. 6. t-SNE visualization results without (left) and with (Right) task-specific fine-tuning. (a) and (b) Visualizations on FineGym99 and FineGym288, respectively. These results show that our task-specific fine-tuning mechanism significantly enlarges the interclass distribution di GLYPH&lt;11&gt; erences.

<!-- image -->

time comparison experiment using four NVIDIA A40 GPUs with a batch size of 24 on the FineGym99 dataset. The results are summarized in Table VII. Despite the inclusion of six loss terms, the epoch training time of ART is 121.8 min, only moderately higher than UniFormerV2 (110.4 min). Moreover, our ART framework achieves a competitive balance between accuracy and speed, with an average inference time

<!-- image -->

(c)Balancebeam: flic-flac with step-out, also with support on one arm

Fig. 7. Visualization of activation maps comparing baseline CAM and regionspecific semantic responses on three action samples of FineGym288. (a) Switch Leap: ART attends to the leading leg and hip, e GLYPH&lt;11&gt; ectively capturing the transition phase that di GLYPH&lt;11&gt; erentiates leap types. (b) Split Jump: Attention concentrates on thigh and foot extensions, modeling the dynamics of leg separation. (c) Flic-Flac With Step-Out (One-Arm Support): ART exhibits partial misalignment, with incomplete tracking of the supporting arm under high-speed motion.

TABLE VII COMPARISON OF COMPUTATIONAL COMPLEXITY, MODEL PARAMETER

## COUNT, AND RECOGNITION ACCURACY

of 36.2 ms / video clip, which is comparable to UniFormerV2 (35.8 ms), while maintaining superior accuracy.

These results highlight ART's ability to achieve a more discriminative spatiotemporal representation with only a marginal increase in complexity. This tradeo GLYPH&lt;11&gt; is particularly favorable in contexts where FGAR accuracy is critical, such as competitive sports analysis, surgical skill assessment, or human-computer interaction systems, where small improve-

Fig. 8. Qualitative visualization of three region-specific tracklets extracted by ART across two fine-grained gymnastics actions. For each video, the top row shows original frames, while the bottom rows overlay regionspecific responses to three semantic queries. Bounding boxes and dashed lines indicate spatial activations and their temporal coherence. ART reliably tracks fine-grained action parts-such as leading legs, gripping hands, and torsional joints-with clear spatiotemporal consistency. (a) Balance Beam: Salto backward stretched-step out. (b) Uneven Bar: Gaint circle backward.

<!-- image -->

ments in accuracy can lead to significant downstream impact. Overall, the comparative analysis confirms that ART is not only e GLYPH&lt;11&gt; ective but also computationally feasible for highprecision video understanding tasks.

## E. Visualization Analysis

In Fig. 7, we visualize the di GLYPH&lt;11&gt; erences between the CAMs [92] generated by the baseline backbone network and the region-specific semantic response maps obtained from ART, across three representative action categories from the FineGym288 dataset. These include: (a) switch leap (leap forward with leg change) , (b) split jump , and (c) flic-flac with step-out, also with support on one arm . In the first two examples of (a) and (b), which involve well-defined and structured body motions, ART demonstrates a clear advantage in tracking temporally coherent, action-relevant regions. Unlike CAM, which typically highlights a coarse blob around the general human figure, ART produces spatially focused and temporally consistent activation over discriminative body parts such as the raised leg or extended thigh. This fine-grained attention e GLYPH&lt;11&gt; ectively captures subtle variations in movement that are critical for distinguishing between visually similar action classes.

In complex high-velocity motions with asymmetric support, ART fails to consistently track the supporting arm, overemphasizing torso and legs, revealing limitations under

rapid deformations or occlusions. Nonetheless, visualizations show that our tracklets generally capture relevant regions and maintain temporal consistency across frames. To validate ART's ability to capture fine-grained dynamics, Fig. 8 shows three region tracklets over time. ART consistently tracks semantically meaningful parts-e.g., legs, torso, and arms in floor gymnastics, or hands, shoulders, and legs on uneven bars-demonstrating improved region localization, temporal alignment, and semantic consistency for modeling subtle action variations.

## V. CONCLUSION

We present a novel discriminative actions tracking framework, namely, ART, built upon Transformers for FGAR. Specifically, given a video consisting of multiple frames and a set of learnable vectors serving as the distinctive region query for each frame, ART first extracts region-specific semantic responses from video frames with distinctive region queries via the self-attention mechanism and then forms a group of action tracklets for better characterizing fine-grained actions. Moreover, to capture diverse semantic responses in each frame and model the correlation of similar response regions across frames, we introduce an MTC-Loss among multiple region responses at spatial, temporal, and tracklet levels. Comprehensive experimental results on four challenging datasets clearly demonstrate the superiority of our proposed ART.
