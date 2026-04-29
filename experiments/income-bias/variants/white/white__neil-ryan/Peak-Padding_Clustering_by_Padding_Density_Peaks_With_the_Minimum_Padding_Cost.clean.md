# Peak-Padding Clustering by Padding Density Peaks With the Minimum Padding Cost

Author: Neil Ryan
University: Auburn University


## Abstract

Clustering complex-shaped clusters is still chal lenging for most existing clustering algorithms. Herein, the peak-padding clustering algorithm (PeakPad)-clustering by padding density peaks with the minimum padding cost-is proposed. PeakPad executes clustering on the density surface and views complex-shaped clusters as combinations of highly associated single-peak clusters. The minimum padding cost that fully considers the surrounding context of a density peak is proposed to reflect a density peak's center potential, enabling PeakPad to have robust center detection performance. Unlike mean-shift (MSC), which detects centers based on their attributes in a complex-shaped density surface embedded in the highdimensional space of density and features, PeakPad detects centers in a standard-shaped surface embedded in the 2-D density-change (DC) density space (composed of density and DC feature). Such standardization allows PeakPad to have fast and robust cluster center detection performance on complexshaped clusters based on the minimum padding cost. Besides, PeakPad can provide a reasonable evaluation of the association between single-peak clusters by using the minimum padding cost. As a result, PeakPad can fast capture complex-shaped clusters, achieve robust center detection performance, and be suitable for large datasets. Benchmark test results on both synthetic and real datasets demonstrate the e GLYPH&lt;11&gt; ectiveness of PeakPad.

Index Terms -Center detection, clustering, complex-shaped clusters, density peak.

## I. INTRODUCTION

C LUSTERING is one of the most fundamental topics in data mining [1]. Its attempt to group similar objects has solidified its irreplaceable role in pattern recognition [2], computer vision [3], image processing [4], biomedical data analysis [22], and so on.

As no consensus on the definition of a cluster has been reached, clustering algorithms based on specific assumptions regarding the nature of a 'cluster' were proposed [5]. As a popular nonparametric clustering technique, mean-shift (MSC)

Digital Object Identifier 10.1109 / TNNLS.2025.3606527

[6] treats data points as an empirical probability density function and views local density maximum areas (density peaks) of the function as centers. It can divide nonspherical clusters by iteratively shifting (assigning) each point to its proximal dense region. However, MSC may overdivide a multipeak cluster (also known as a multicenter cluster) [7], since it views each density peak as a center.

In 2014, Science published the clustering method called density peak clustering (DPC) [8]. DPC addresses the overdivision problem of MSC by finding density peaks with high 'center potential' as cluster centers. Based on DPC's assumption, a point's center potential ( GLYPH&lt;13&gt; = GLYPH&lt;26&gt; GLYPH&lt;1&gt; GLYPH&lt;14&gt; ) depends on two factors: GLYPH&lt;26&gt; (local density) and GLYPH&lt;14&gt; (the minimum distance to a higher density point). DPC uses a decision graph (i.e., a GLYPH&lt;26&gt; -GLYPH&lt;14&gt; plot) to select centers. Once the centers are chosen, each remaining point is assigned to the cluster of its nearest higher density point.

Although DPC's center detection is e GLYPH&lt;11&gt; ective and easy to implement, its Euclidean-distance-based GLYPH&lt;14&gt; struggles with complex-shaped clusters [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19]. Cheng et al. [19] used granular balls (GBs), defined their GLYPH&lt;26&gt; and GLYPH&lt;14&gt; , and clustered them via DPC, significantly reducing runtime. Guan et al. [18] introduced 'main' and 'satellite' peaks based on density and designed a GLYPH&lt;14&gt; -based attenuator to suppress satellite peaks, yielding a more robust decision graph. Many works have also explored alternative distance or dissimilarity measures. For example, Liu et al. [9] used a shared-nearest-neighbor-based GLYPH&lt;14&gt; to capture centers in sparse clusters; Guan et al. [10] proposed a density-deviationcost GLYPH&lt;14&gt; to quickly identify main peaks in complex-shaped clusters; Sharma et al. [16] designed a mixture-distancebased GLYPH&lt;14&gt; for mixed data, using a divergence-based nonlinear S-distance [20] to better detect complex boundaries [21]; Pizzagalli et al. [11] kept DPC's GLYPH&lt;14&gt; but adopted a shortestpath strategy for more robust shape reconstruction. Despite their improvements, these methods still follow DPC's unrobust GLYPH&lt;13&gt; -based center detection idea.

DPC's clustering process can be interpreted as a peakpadding strategy, referred to as 'the DPC-padding strategy': each point is padded (assigned) to a higher density area based on its padding cost GLYPH&lt;13&gt; , and peaks with relatively large GLYPH&lt;13&gt; values are selected as centers , as illustrated in Fig. 1 (upper). For example, peak p 0 has a much higher padding cost GLYPH&lt;13&gt; (shown as the red rectangle) than point x 0, making it a more suitable center candidate. However, GLYPH&lt;13&gt; covers and extends well beyond the valley region-i.e., the area representing the minimum padding cost-and therefore fails to precisely reflect the valley structure, which is key to identifying cluster boundaries [6]. To address this, we propose the min-padding strategy: points

Fig. 1. DPC-padding strategy and the proposed min-padding strategy on a continuous density curve. Notably, nonpeak point x 0 on a continuous curve has no padding cost, since its nearest higher density point x 0 0 is infinitely close to it, i.e., GLYPH&lt;14&gt; x 0 = jj x 0 GLYPH&lt;0&gt; x 0 0 jj! 0. Then, GLYPH&lt;13&gt; x 0 = GLYPH&lt;26&gt; x 0 GLYPH&lt;1&gt; GLYPH&lt;14&gt; x 0 ! 0.

<!-- image -->

are padded using their minimum padding cost GLYPH&lt;18&gt; , and peaks with relatively large GLYPH&lt;18&gt; values are selected as centers , as shown in Fig. 1 (bottom). This approach captures the valley structure more accurately.

Unlike the global GLYPH&lt;13&gt; metric, which often stretches beyond the actual density valleys (see Fig. 1, top), GLYPH&lt;18&gt; closely follows local density minima, turning hidden valley structures into clear, measurable features. This allows for direct detection of saddle points-something the GLYPH&lt;13&gt; metric cannot do. As shown in Fig. 1 (bottom), our method removes the overextension issue of DPC's GLYPH&lt;13&gt; at cluster boundaries, producing cluster borders that align well with the true density valleys and providing a solid basis for analyzing complex data distributions.

Fig. 2 compares the performance of the DPC-padding strategy and the min-padding strategy on 1-D continuous density curves. As shown in Fig. 2(a), the DPC-padding strategy produces identical decision graphs for tasks i and ii . However, for ideal clustering-cutting at lower density valleys-centers should be p 1 and p 3 in task i , and p 2 and p 3 in task ii . This shows that the GLYPH&lt;13&gt; attribute does not reliably guide correct center selection. In contrast, the proposed min-padding strategy, by explicitly considering the valley structure [Fig. 2(b)], easily solves both tasks. This aligns better with human perception, as our nervous system tends to favor economical solutions [23], illustrated in Fig. 2(c).

The e GLYPH&lt;11&gt; ectiveness of a padding strategy heavily depends on the choice of dissimilarity measure, and 'Euclidean distance' is not suitable in this context. Peaks near wide valleys may have larger padding costs than those near deeper valleys when measured by Euclidean distance, despite deep valleys better representing cluster boundaries [10]. In Fig. 3, peak p 2 has a lower minimum padding cost than p 1, causing the min-padding strategy to yield a suboptimal result by padding peak p 2. Moreover, accurately calculating the minimum padding cost in Euclidean space is di GLYPH&lt;14&gt; cult, requiring a specific discrete summation that is time-consuming and prone to unavoidable computational errors.

To address the limitations mentioned above, we propose a new dissimilarity measure for the min-padding strategy called the ' density change distance (DC distance) .' Building on this, the Peak-Padding Clustering algorithm (PeakPad) 1 is introduced. The main contributions of PeakPad are as follows.

- 1) The DC-distance-based minimum padding cost e GLYPH&lt;11&gt; ectively reflects the center potential of peaks and can be computed e GLYPH&lt;14&gt; ciently, enabling PeakPad to achieve robust and high-performance center detection.
- 2) PeakPad treats each density peak as representing a single-peak cluster. By appropriately padding noncenter peaks into final multipeak clusters, it can naturally capture complex-shaped multipeak clusters.
- 3) PeakPad uses kNN distances as input, making it wellsuited for clustering large-scale datasets.

The rest of the article is organized as follows. Section II introduces related works; Section III focuses on the proposed PeakPad algorithm; Section IV presents benchmark tests and discussions; and Section V concludes the article.

## II. RELATED WORKS

MSC [6] identifies cluster centers as dense regions in the feature space. It models data points as an empirical probability density function using a specific kernel function and locates the local maxima of this function, which correspond to dense regions. Each point iteratively performs an 'MSC' procedure-a gradient ascent on the local density surface-until convergence. The shifted points then form groups, while the stationary points (local maxima) are identified as cluster centers, thus completing the clustering process. Although MSC e GLYPH&lt;11&gt; ectively captures clusters of arbitrary shapes without requiring the number of clusters to be specified in advance, its tendency to treat every local density maximum as a center can lead to overdivision in multipeak clusters [7].

Although DPC [8] builds on the core idea of MSC, it di GLYPH&lt;11&gt; ers by not treating all density peaks as centers. Instead, DPC selects centers based on its center assumption-that cluster centers are density peaks with both high density and relatively large distance from points of higher density. For clustering, DPC assigns each noncenter point to the same cluster as its nearest neighbor with higher density, a process known as DPC's allocation strategy . Typically, DPC is viewed as a hierarchical clustering method [25] employing a specific linkage metric called the minimum center-boosting distance [25], which aligns with the proposed Peak-Padding strategy .

Nevertheless, DPC's allocation strategy often fails to accurately reconstruct complex shapes, as it may wrongly associate unrelated points [11]. Furthermore, as mentioned in the Introduction, DPC's center detection based on the GLYPH&lt;13&gt; metric (referred to as GLYPH&lt;13&gt; -based center detection) is unreliable for multipeak clusters. Although many methods [9], [10], [11], [12], [13], [14], [15], [16], [17], [18] have been proposed to improve center detection, they still depend on DPC's original GLYPH&lt;13&gt; -based approach and lack robustness for multipeak clusters. In contrast, PeakPad o GLYPH&lt;11&gt; ers robust center detection for complex-shaped clusters by adopting a new center assumption: cluster centers are density peaks with relatively large minimum padding costs.

DBSCAN [26], a classic density-based clustering method, considers a cluster as a set of density-connected points. The 'density-connected' is defined using GLYPH&lt;15&gt; and minPts parameters. Although DBSCAN can reconstruct complex-shaped clusters

1 The code is available at https: // 
## III. PEAK-PADDING CLUSTERING (PEAKPAD)

This section provides a detailed introduction to the proposed PeakPad algorithm. Fig. 4 illustrates its clustering process, which consists of two main steps: 1) building the peak graph and 2) performing peak-padding clustering.

Step 1: Given the dataset X and parameter k , PeakPad first estimates the density of data points [using (1)], then detects density peaks, single-peak clusters, border points, and saddle points (according to Definitions 1 to 4). Using the adjacency relationships among single-peak clusters, PeakPad constructs the peak graph.

Step GLYPH&lt;24&gt; 2: Based on the peak graph, PeakPad iteratively pads peaks to adjacent higher density peaks by minimizing the padding cost GLYPH&lt;18&gt; [as defined in (3)] while simultaneously generating a dendrogram. Next, it selects the top nc peaks with the largest GLYPH&lt;18&gt; values as cluster centers via a clear decision graph, following our center Assumption 1. The dendrogram is then cut into nc peak groups, and finally, clustering is completed by grouping all points within the same peak group into clusters.

## A. Peak Graph Building

1) Detection of Peaks: Given a dataset X = f x 1 ; x 2 ; : : : ; xn j xi 2 R d g , a kNN-based density function GLYPH&lt;26&gt; : R d ! R + is applied to estimate point xi 's density GLYPH&lt;26&gt; xi , as follows:

<!-- formula-not-decoded -->

where Nk ( xi ) indicates the set of xi 's k nearest neighbors.

On this basis, a density peak is viewed as the density maximum point within its k nearest neighbors [24], as follows.

Definition 1: If GLYPH&lt;26&gt; xi &gt; max xj 2 Nk ( xi ) ( GLYPH&lt;26&gt; xj ), then xi is a density peak p .

According to Definition 1, the dataset X is divided into peaks P and nonpeak points ¯ P , i.e., X = P [ ¯ P , where P = f p 1 ; p 2 ; : : : ; pnp g indicates all np peaks of dataset X .

Based on the clustering idea of MSC [6], each nonpeak point xi 2 ¯ P is associated with its nearest neighbor xj within a higher density area, i.e., xj = arg min x dxxi , s.t. x 2 Nk ( xi ) ; GLYPH&lt;26&gt; x &gt; GLYPH&lt;26&gt; xi . As a result, each nonpeak point will be associated with a peak. Then, a single-peak cluster is defined as follows.

Definition 2: A single-peak cluster, denoted as S ( p ), is composed of one peak p and all nonpeak points associated with the peak.

Therefore, peak p is considered as the representative point (center) of S ( p ).

2) Graph Building: Peaks P is built into a peak graph G ( P ; E ), where E = f epi p j j9 Bpipj g . If border point(s) Bpipj exist(s) between the single-peak clusters S ( pi ) and S ( pj ), then an edge epi p j connects pi and pj in the peak graph G .

Border points are considered to only exist on both sides of the borderline between intersecting single-peak clusters. That is to say, the borderline divides the mutual-proximal border points of di GLYPH&lt;11&gt; erent single-peak clusters. Therefore, to e GLYPH&lt;11&gt; ectively detect the mutual-proximity between border points, a small-value kb is introduced

Fig. 4. Clustering idea of the proposed PeakPad algorithm.

<!-- image -->

where parameter k and '2' are used to prevent kb from becoming too small to detect border points, and bGLYPH&lt;1&gt;c is a floor function.

Then, border points Bpipj are defined as follows:

Definition 3: If mutual-proximity points xa and xb are in di GLYPH&lt;11&gt; erent single-peak clusters, i.e., xa 2 S ( pi ) \ Nkb ( xb ) ; xb 2 S ( pj ) \ Nkb ( xa ), then, the low density point between xa and xb is a border point b between peaks pi and pj , i.e., b = arg min x 2f xa ; xb g ( GLYPH&lt;26&gt; x ) ; b 2 Bpipj . This ensures that the border point has a lower density than either of the associated density peaks.

Besides, a saddle point is defined as follows.

Definition 4: If border point b has the highest density in its border set B , then point b is a saddle point, denoted as b GLYPH&lt;3&gt; , i.e., b GLYPH&lt;3&gt; = arg max b 2 B ( GLYPH&lt;26&gt; b ).

According to Definition 3, border points can be readily identified. If two single-peak clusters share border points, their corresponding peaks are connected in the peak graph. As shown in Fig. 4 (Step 1: peak graph building), peak areas that share border points are linked in the peak graph.

In what follows, a detailed description of the min-padding strategy on peak graph G will be given.

## B. Min-Padding Strategy

The min-padding strategy is a bottom-up hierarchical clustering technique applied to peak graphs [13].

- 1) Initially, we have a peak graph G containing np peaks. Consider G [ t ] = G ( P [ t ] ; E [ t ] ) as the peak graph after t padding operations, where t 2 f 0 ; 1 ; : : : ; np GLYPH&lt;0&gt; 1 g , and P [ t ] = f p [ t ] 1 ; p [ t ] 2 ; : : : ; p [ t ] n [ t ] p g is the set of peaks at time t .
- 2) In each padding operation, we select the peak p [ t ] i with the global minimum padding cost and pad it into its adjacent higher density peak p [ t ] j . The original peak p [ t ] i is then removed from P [ t ] , updating the peak set to P [ t + 1] = P [ t ] n f p [ t ] i g .
- 3) Simultaneously, p [ t + 1] j inherits all existing connections in E [ t ] from both peaks p [ t ] i and p [ t ] j to other peaks, updating the edge set to E [ t + 1] . The peak graph is then updated as G [ t + 1] = G ( P [ t + 1] ; E [ t + 1] ).
- 4) Update t t + 1 and repeat the process until j P [ t ] j = 1 (i.e., the peak graph reduces to a single peak), completing the padding task.

During the padding process, peak pi can obtain the minimum padding cost, as follows:

<!-- formula-not-decoded -->

where V p [ t ] i p [ t ] j is a valley between adjacent peaks pi and pj within peak graph G [ t ] (i.e., 9 e p [ t ] i p [ t ] j 2 E [ t ] ), and Cost( GLYPH&lt;1&gt; ) is a padding cost function.

Equation (3) implies that a peak is only allowed to be padded to an adjacent higher density peak within its connected component. Therefore, if peak pi is the highest density peak of a connected component in peak graph G , then, peak p [ t ] i ( 8 t ) has no adjacent higher density peak to satisfy its padding operation, i.e., GLYPH&lt;18&gt; pi = ? . Consequently, peak pi must serve as a cluster center, referred to as a 'center peak.'

For each center peak pi with GLYPH&lt;18&gt; pi = ? , its padding cost is assigned as follows:

<!-- formula-not-decoded -->

where the scaling factor GLYPH&lt;12&gt; &gt; 1 (default: 2) ensures that center peaks stand out clearly in the GLYPH&lt;18&gt; -decision graph. Under the minpadding strategy, center peaks are iteratively padded into the global maximum-density peak.

## C. Center Assumption

Based on the min-padding strategy, our center assumption naturally emerges.

Assumption 1: Cluster centers are density peaks with a relatively large minimum padding cost GLYPH&lt;18&gt; value.

In PeakPad, a cluster center can be regarded as the representative point of a cluster, as it corresponds to the highest density point within that cluster.

Fig. 5. Idea of DC distance based on DC cost.

<!-- image -->

Based on Assumption 1, a decision graph-similar to that used in DPC [8]-can be constructed to assist in the manual selection of cluster centers without requiring prior knowledge. Once the cluster centers are identified, PeakPad performs clustering by iteratively padding the remaining noncenter peaks to centers based on the minimum padding cost. Alternatively, the number of clusters can be preset, and the corresponding number of peaks with the highest padding costs can be selected as centers. Clustering is then completed by assigning each peak's cluster label to its corresponding single-peak cluster.

A detailed explanation of the 'DC distance'-a dissimilarity measure used for estimating GLYPH&lt;18&gt; -will be provided in Section III-D.

## D. Density Change Distance

1) DC distance of a Path: Consider a compact path GLYPH&lt;0&gt; xi xj = f x [0] ; x [1] ; : : : ; x [ n GLYPH&lt;0&gt; ] g between points xi and xj , satisfying that adjacent points along the path are neighbors, where x [0] = xi and x [ n GLYPH&lt;0&gt; ] = xj . The density change (DC) cost of path GLYPH&lt;0&gt; xi xj is defined as follows.

Definition 5: For a path GLYPH&lt;0&gt; xi xj between points xi and xj , its DC cost is the sum of relative DCs of points along path GLYPH&lt;0&gt; xi xj , as follows.

<!-- formula-not-decoded -->

where GLYPH&lt;1&gt; GLYPH&lt;26&gt; x [ m ] = GLYPH&lt;26&gt; x [ m ] GLYPH&lt;0&gt; GLYPH&lt;26&gt; x [ m GLYPH&lt;0&gt; 1] represents the density change from points x [ m ] to x [ m GLYPH&lt;0&gt; 1], and j GLYPH&lt;1&gt; GLYPH&lt;26&gt; x [ m ] j = j GLYPH&lt;26&gt; x [ m ] GLYPH&lt;0&gt; GLYPH&lt;26&gt; x [ m GLYPH&lt;0&gt; 1] j denotes the absolute DC .

Let GLYPH&lt;21&gt; = max fj GLYPH&lt;1&gt; GLYPH&lt;26&gt; x [1] j ; j GLYPH&lt;1&gt; GLYPH&lt;26&gt; x [2] j ; : : : ; j GLYPH&lt;1&gt; GLYPH&lt;26&gt; x [ n GLYPH&lt;0&gt; ] jg . Since dataset X can be assumed to be sampled from an underlying continuous and smooth probability density surface, then a continuous path ˜ GLYPH&lt;0&gt; xi xj between xi and xj on the surface can be obtained by letting GLYPH&lt;21&gt; ! 0 (i.e., n GLYPH&lt;0&gt; !1 ). Then, the DC distance is defined as follows.

Definition 6: For a continuous path ˜ GLYPH&lt;0&gt; xi xj , its DC cost is defined as its DC distance

<!-- formula-not-decoded -->

Fig. 5 illustrates the idea of DC distance based on the DC cost. As shown, when n GLYPH&lt;0&gt; !1 , path GLYPH&lt;0&gt; xa xb ! ˜ GLYPH&lt;0&gt; xa xb . Meanwhile, DC cost ( GLYPH&lt;0&gt; xi xj ) ! DC ( ˜ GLYPH&lt;0&gt; xi xj ).

Due to the existence of j GLYPH&lt;1&gt; j in (6), DC ( ˜ GLYPH&lt;0&gt; xi xj ) cannot be expressed in definite integral form [30]. To solve this, ˜ GLYPH&lt;0&gt; is considered as a combination of density-change monotone paths (denoted as ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; ) as follows:

<!-- formula-not-decoded -->

where x (0) = xi ; x ( nz + 1) = xj , and x (1) ; x (2) ; ::; x ( nz ) 2 ˜ GLYPH&lt;0&gt; xi xj are all local density extremum points along path ˜ GLYPH&lt;0&gt; xi xj , i.e., x (1) ; x (2) ; : : : ; x ( nz ) on the path that are either local density minima or local density maxima.

Along each density-change monotone segment ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; x ( z ) x ( z + 1) = f x [0] ; x [1] ; : : : ; x [ n GLYPH&lt;0&gt; ( z ) ] g , where x [0] = x ( z ) and x [ n GLYPH&lt;0&gt; ( z ) ] = x ( z + 1), the DC GLYPH&lt;1&gt; GLYPH&lt;26&gt; x [ m ] will maintain the same sign. Thus, the DC distance of ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; x ( z ) x ( z + 1) can be calculated as follows:

<!-- formula-not-decoded -->

Therefore, (6) can be rewritten as follows:

<!-- formula-not-decoded -->

Particularly, if ˜ GLYPH&lt;0&gt; xi xj is a density-change monotone path, i.e., ˜ GLYPH&lt;0&gt; xi xj = ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; xi xj , then the DC distance between xi and xj is a fixed value, as follows:

<!-- formula-not-decoded -->

Equation (10) can work for a single-peak cluster because there must be a ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; path between any two points within a single-peak cluster, see Fig. 6(a).

2) DC Distance Between Adjacent Peaks: Consider density peaks pi ; pj 2 P are in two intersecting single-peak clusters. Therefore, a path ˜ GLYPH&lt;0&gt; pi pj can be split into two density-change monotone paths, as ˜ GLYPH&lt;0&gt; pi pj = ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; pib [ ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; bpj , where point b can be any point of Bpipj between single-peak clusters, and 8 b 2 Bpipj ; GLYPH&lt;26&gt; b &lt; min( GLYPH&lt;26&gt; pi ; GLYPH&lt;26&gt; pj ). Then, for each border point b , a path with DC distance DC ( ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; pib [ ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; bpj ) can be obtained as follows:

<!-- formula-not-decoded -->

herein, the DC distance between two adjacent peaks is defined as follows.

Fig. 6. (a) DC distance within a single-peak cluster, and (b) DC distance between peaks and the DC width of a valley.

<!-- image -->

Definition 7: For density peaks pi and pj within two intersecting single-peak clusters, their DC distance DC ( pi ; pj ) is defined as the DC distance of the shortest path between them

<!-- formula-not-decoded -->

With saddle b GLYPH&lt;3&gt; = arg max b 2 Bp i p j ( GLYPH&lt;26&gt; b ), we have

<!-- formula-not-decoded -->

Let GLYPH&lt;26&gt; pi &lt; GLYPH&lt;26&gt; pj , the DC width W ( Vpipj ) of valley Vpipj is the horizontal DC distance from peak pi to peak pj 's peak area, denoted as P area ( pj ) (i.e., the local density area leading by peak pj ). In other words, W ( Vpipj ) is the DC distance between peak pi and its nearest same-density point x ( pi ) 2 P area ( pj ), where GLYPH&lt;26&gt; pi = GLYPH&lt;26&gt; x ( pi ). Therefore, according to (13), we have

<!-- formula-not-decoded -->

Equations (13) and (14) indicate that the DC distance and the valley DC width between adjacent density peaks can be obtained as long as the saddle point b GLYPH&lt;3&gt; is found in the borderline Bpipj , as shown in Fig. 6(b).

3) DC Surface Path Expression: Given a path ˜ GLYPH&lt;0&gt; xi xj , its density surface path is defined as in (15). With GLYPH&lt;0&gt; xi xj 2 R d , GLYPH&lt;8&gt; ( ˜ GLYPH&lt;0&gt; xi xj ) 2 R d + 1

<!-- formula-not-decoded -->

Given a function GLYPH&lt;24&gt; : R d ! R 1 satisfies 8 xa ; xb 2 ˜ GLYPH&lt;0&gt; xi xj ; j GLYPH&lt;24&gt; xa GLYPH&lt;0&gt; GLYPH&lt;24&gt; xb j = DC ( xa ; xb ). Path GLYPH&lt;24&gt; ( ˜ GLYPH&lt;0&gt; xi xj ) as the mapping of ˜ GLYPH&lt;0&gt; xi xj in DC space is called a DC path . Then, the DC surface path of ˜ GLYPH&lt;0&gt; xi xj is defined as follows:

<!-- formula-not-decoded -->

Notably, mapping GLYPH&lt;8&gt; ( ˜ GLYPH&lt;0&gt; xi xj ) to GLYPH&lt;10&gt; ( ˜ GLYPH&lt;0&gt; xi xj ), i.e., mapping density surface path to DC surface path, can be considered as a dimension reduction operation.

As known, path ˜ GLYPH&lt;0&gt; xi xj can be arbitrary shapes in feature space, so density surface path GLYPH&lt;8&gt; ( ˜ GLYPH&lt;0&gt; xi xj ) can also own arbitrary shapes. But what is fascinating is that DC surface path

GLYPH&lt;10&gt; ( ˜ GLYPH&lt;0&gt; xi xj ) , GLYPH&lt;0&gt; GLYPH&lt;24&gt; ( ˜ GLYPH&lt;0&gt; xi xj ) ; GLYPH&lt;26&gt; ( ˜ GLYPH&lt;0&gt; xi xj ) GLYPH&lt;1&gt; can enjoy a standard shape (that a function combination can express), since a functional relationship exists between GLYPH&lt;24&gt; ( GLYPH&lt;1&gt; ) and GLYPH&lt;26&gt; ( GLYPH&lt;1&gt; ), as will be shown later.

Given a density-change monotone path ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; xi xj , according to (10), its DC path GLYPH&lt;24&gt; ( ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; xi xj ) is a fixed line segment with length DC ( xi ; xj ) = j ln( GLYPH&lt;26&gt; xi =GLYPH&lt;26&gt; xj ) j = j GLYPH&lt;24&gt; xi GLYPH&lt;0&gt; GLYPH&lt;24&gt; xj j between GLYPH&lt;24&gt; xi and GLYPH&lt;24&gt; xj in DC space. Then, the mapping of path ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; xi xj is a line segment at [ GLYPH&lt;24&gt; xi ; GLYPH&lt;24&gt; xj ] in DC space, where 8 x 2 ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; xi xj , its mappin GLYPH&lt;24&gt; x in DC space is

<!-- formula-not-decoded -->

where function GLYPH&lt;31&gt; ( GLYPH&lt;1&gt; ) 2 fGLYPH&lt;0&gt; 1 ; 0 ; 1 g returns the monotonicity of path ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; as in the following equation:

<!-- formula-not-decoded -->

where GLYPH&lt;31&gt; ( ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; ) = 1 means that ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; is a density-boosting path. Therefore, for path ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; xi xj , the functional relation between GLYPH&lt;24&gt; ( GLYPH&lt;1&gt; ) and GLYPH&lt;26&gt; ( GLYPH&lt;1&gt; ) can be described as follows:

<!-- formula-not-decoded -->

According to (7), any shaped path ˜ GLYPH&lt;0&gt; xi xj can be viewed as a combination of monotone ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; paths, i.e., ˜ GLYPH&lt;0&gt; xi xj = S nz z = 0 ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; x ( z ) x ( z + 1) . Then, its DC surface path function f ( GLYPH&lt;1&gt; ) is defined as follows:

<!-- formula-not-decoded -->

where GLYPH&lt;24&gt; x (1) ; GLYPH&lt;24&gt; x (2) ; : : : ; GLYPH&lt;24&gt; x ( nz ) are the mappings of all local density extremum points x (1) ; x (2) ; : : : ; x ( nz ) in DC space, and GLYPH&lt;24&gt; x (0) GLYPH&lt;24&gt; xi ; GLYPH&lt;24&gt; x ( nz + 1) GLYPH&lt;24&gt; xj . According to (20), we get the following. Theorem 1: For paths ˜ GLYPH&lt;0&gt; xa xb = S nz z = 0 ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; x ( z ) x ( z + 1) and ˜ GLYPH&lt;0&gt; xc xd = S nz z = 0 ˜ GLYPH&lt;0&gt; GLYPH&lt;14&gt; x 0 ( z ) x 0 ( z + 1) , if GLYPH&lt;26&gt; x ( m ) = GLYPH&lt;26&gt; x 0 ( m ) , m = f 0 ; 1 ; : : : ; nz + 1 g , then, DC surface paths GLYPH&lt;10&gt; ( ˜ GLYPH&lt;0&gt; xa xb ) and GLYPH&lt;10&gt; ( ˜ GLYPH&lt;0&gt; xc xd ) have the same shape.

Inspired by Theorem 1, arbitrary-shaped density surface paths are transformed into DC surface paths to achieve a shape-standardization of density surface paths. On this basis, the DC-based minimum padding cost GLYPH&lt;18&gt; of peaks can be easily calculated.

Fig. 7. Calculation process of our DC-based minimum padding cost.

<!-- image -->

4) DC-Based Minimum Padding Cost: Valley Vpipj is a DC surface path GLYPH&lt;10&gt; ( ˜ GLYPH&lt;0&gt; pi x ( pi )), where x ( pi ) 2 P area ( pj ) and GLYPH&lt;26&gt; pi = GLYPH&lt;26&gt; x ( pi ) . According to (20), we have valley Vpipj 's DC surface path function as follows:

<!-- formula-not-decoded -->

According to (21), function f ( GLYPH&lt;24&gt; x ) is about axis GLYPH&lt;24&gt; x = GLYPH&lt;24&gt; b GLYPH&lt;3&gt; symmetry. Therefore, the area of Vpipj is the enclosed area composed of function f ( GLYPH&lt;24&gt; x ) and horizontal line f 0( GLYPH&lt;24&gt; x ) = GLYPH&lt;26&gt; pi on interval GLYPH&lt;24&gt; x 2 [ GLYPH&lt;24&gt; pi ; GLYPH&lt;24&gt; x ( pi )], as follows:

<!-- formula-not-decoded -->

Fig. 7 shows the calculation process of a peak's (i.e., point x (3)) DC-based minimum padding cost. As shown, a peak's DC-based minimum padding cost can be fast calculated, as long as the saddle point toward the adjacent higher density peak is found. More importantly, our DC-based minimum padding cost gives a deep valley a large cost, which ensures that PeakPad e GLYPH&lt;11&gt; ectively captures the deep valleys between clusters.

Let GLYPH&lt;24&gt; pi = 0, we have the interval [ GLYPH&lt;24&gt; pi ; GLYPH&lt;24&gt; b GLYPH&lt;3&gt; ] become [0 ; DC ( pi ; b GLYPH&lt;3&gt; )] = [0 ; ln( GLYPH&lt;26&gt; pi =GLYPH&lt;26&gt; b GLYPH&lt;3&gt; )]. Hence, area Cost( Vpipj ) can be calculated as follows:

<!-- formula-not-decoded -->

According to (3), the DC-based minimum padding cost is calculated as follows:

<!-- formula-not-decoded -->

9 b GLYPH&lt;3&gt; 2 B p [ t ] i p [ t ] j means that there is saddle point b GLYPH&lt;3&gt; between single-peak clusters S ( p [ t ] i ) and S ( p [ t ] j ) after t padding operations. This constraint is equivalent to 9 e p [ t ] i p [ t ] j 2 E [ t ] as in (3), since E = f epi p j j9 Bpipj g .

5) Hierarchical Features: During clustering, every remaining noncenter peak p [ t ] i leads a single-peak cluster S ( p [ t ] i ). As a bottom-up hierarchical clustering technique, PeakPad views the padding cost between single-peak clusters as their dissimilarity, therefore, its linkage metric is defined as follows:

<!-- formula-not-decoded -->

Equation (25) tells that the dissimilarity between two singlepeak clusters only depends on the density values of their saddle point and the low-density peak. Based on this, the dissimilarity between single-peak clusters can be fast evaluated by fastfinding saddle points.

Fig. 8 illustrates the clustering process of the PeakPad algorithm on a toy dataset with three clusters. The PeakPad algorithm identifies four peak areas, each guided by a density peak ( A GLYPH&lt;24&gt; D ), and detects boundary areas between peak areas clusters, where the highest density point is the saddle point. If two peak areas share boundary points, they are considered connected in the peak graph.

Next, peaks are merged based on their minimum padding cost. Peak B , with the lowest cost, is first padded into peak C 's area (with GLYPH&lt;18&gt; B = 2 GLYPH&lt;1&gt; GLYPH&lt;26&gt; B GLYPH&lt;1&gt; (ln( GLYPH&lt;26&gt; B =GLYPH&lt;26&gt; b GLYPH&lt;3&gt; 1 ) + ( GLYPH&lt;26&gt; b GLYPH&lt;3&gt; 1 =GLYPH&lt;26&gt; B ) GLYPH&lt;0&gt; 1) GLYPH&lt;25&gt; 0 : 0137), and removed from the list of peaks. Peak A , with a higher cost, is then padded into peak C 's area (with GLYPH&lt;18&gt; A = 2 GLYPH&lt;1&gt; GLYPH&lt;26&gt; A GLYPH&lt;1&gt; (ln( GLYPH&lt;26&gt; A =GLYPH&lt;26&gt; b GLYPH&lt;3&gt; 2 ) + ( GLYPH&lt;26&gt; b GLYPH&lt;3&gt; 2 =GLYPH&lt;26&gt; A ) GLYPH&lt;0&gt; 1) GLYPH&lt;25&gt; 1 : 3073), and removed. Peaks C and D remain independent, as they have no shared boundaries, and are identified as the center peaks.

To highlight center peaks C and D in the decision graph, we set their padding cost to twice the maximum existing padding cost (i.e., GLYPH&lt;18&gt; C = GLYPH&lt;18&gt; D = 2 GLYPH&lt;1&gt; max( GLYPH&lt;18&gt; A ; GLYPH&lt;18&gt; B ) = 2 : 6146). By observing the decision graph, we can easily identify peaks A , C , and D as the final cluster centers based on their high padding costs GLYPH&lt;18&gt; (while peak B is padded to peak C ), and thereby obtain the clustering result.

## E. Pseudocode and Complexity

Algorithms 1 and 2 show the pseudocode of the proposed PeakPad algorithm in two steps: 1) peak graph building and 2) peak-padding clustering.

Peak graph building (step 1) needs a computational complexity of O ( n log( n ) + n ˜ k + n ), where ˜ k means that a point's

Fig. 8. Clustering process of the PeakPad algorithm on a toy dataset.

<!-- image -->

## Algorithm 1 PeakPad-Step 1: Peak Graph Building

```
Input: Dataset X = f x 1 ; x 2 ; : : : ; xn g , neighborhood parameter k Output: Peak graph G ( P ; E ), single-peak clusters S ( P ) = f S ( p ) j p 2 P g 1 Compute kNN distances using fast k NN technique [31] 2 Estimate the density values GLYPH<26> ( X ) = f GLYPH<26> xi j xi 2 X g using Eq. (1) // Eq. (1): Density estimation 3 Initialize peak set: P X 4 Initialize data label: 5 for all points xi 2 X do 6 Label ( xi ) unique value 7 Set flag: isPeak ( xi ) 1 // Binary peak status 8 end for 9 Sort X by density descending: X 0 sort ( X ; GLYPH<26> ) 10 for all points xi 2 X 0 (high to low density) do 11 for all neighbors xj 2 Nk ( xi ) (near to far) do 12 if GLYPH<26> xj > GLYPH<26> xi then 13 Update status: isPeak ( xi ) 0 (see Definition 1)) 14 Remove peak: P P n f xi g 15 Propagate label: Label ( xi ) Label ( xj ) 16 break neighbor loop 17 end if 18 end for 19 end for 20 Form single-peak clusters: S ( P ) ff x j Label ( x ) = p g j p 2 P g (see Definition 2) 21 for all peak pairs ( pi ; pj ) 2 P × P do 22 if a boundary connection Bpipj exists (see Definition 3) then 23 Add edge: E E [ f epi p j g 24 end if 25 end for 26 return G ( P ; E ) and S ( P )
```

nearest higher density point is its ˜ k th neighbor (an average concept). Actually, most points can find a really close higher density point, i.e., ˜ k GLYPH&lt;28&gt; k . Then, peak-padding clustering (step 2) needs a complexity of O ( n 2 p + np ).

Therefore, the overall computational complexity of PeakPad is O ( n log( n ) + n ˜ k + n 2 p ), with ˜ k ; np GLYPH&lt;28&gt; n .

## Algorithm 2 PeakPad-Step 2: Peak-Padding Clustering

- Input: Dataset X , peak set P , peak graph G ( P ; E ), singlepeak clusters S ( P ) = f S ( p ) j p 2 P g Output: Final clustering result C = f C 1 ; C 2 ; : : : ; Cnc g 1 Initialize iteration counter: t 0 2 Initialize peak set: P [ t ] P 3 Initialize peak graph: G [ t ] G ( P ; E ) 4 while j P [ t ] j &gt; 1 do 5 for each peak pi 2 P [ t ] do 6 Compute minimum padding cost GLYPH&lt;18&gt; pi via Eq. (24) // Eq. (24): Padding cost definition . 7 end for 8 Record the max padding cost in P [ t ] : GLYPH&lt;18&gt; max = max p 2 P [ t ] GLYPH&lt;18&gt; p 9 for each peak ˆ p 2 P [ t ] with GLYPH&lt;18&gt; ˆ p = ? do 10 Set GLYPH&lt;18&gt; ˆ p GLYPH&lt;12&gt; GLYPH&lt;1&gt; GLYPH&lt;18&gt; max , with the default value GLYPH&lt;12&gt; = 2. // Eq. (4): Padding cost for center peaks 11 end for 12 Select optimal peak to pad: p GLYPH&lt;3&gt; arg min p 2 P [ t ] GLYPH&lt;18&gt; p 13 Update peak set: P [ t + 1] P [ t ] n f p GLYPH&lt;3&gt; g 14 Increment counter: t t + 1 15 Rebuild peak graph: G [ t ] 16 end while 17 Manually select peaks with top GLYPH&lt;18&gt; values as cluster centers from decision graph 18 Pad (assign) remaining non-center peaks into higherdensity peaks with the minimum padding cost 19 Associate peaks (representing single-peak clusters) to form final clusters 20 return Clustering result C = f C 1 ; C 2 ; : : : ; Cnc g

```
IV. BENCHMARK TESTS
```

## A. Test Set Up

Datasets: Fifteen commonly used synthetic datasets and thirteen popular real-world datasets are selected to benchmark the proposed clustering algorithm, as shown in Table I. Additionally, 160 synthetic datasets with varying geometric structures and dimensions-including 80 Gaussian sphere datasets and 80 ellipsoidal cluster datasets [54]-are used to evaluate the cluster center detection performance.

Comparison algorithms and settings: Four classic clustering algorithms ( K -means [47], DBSCAN [26], MSC [6], and self-tuning spectral clustering (SSC) [33]), several recent

Fig. 9. Results of di GLYPH&lt;11&gt; erent algorithms on synthetic datasets.

<!-- image -->

TABLE I DATASETS SUMMARY

DPC-based methods (DPC [8], SSSP-DPC [11], SNN-DPC [9], UP-DPC [17], DBSCAN-DPC [34], GB-DP [19], and TC [50]), and the proposed PeakPad are presented for comparison. The parameters of algorithms are set according to their best performance over a large range of possible configurations. For K -means and SSC, the best results among ten runs are selected; while for DPC-based algorithms, appropriate density peaks are selected as cluster centers by observing decision graphs.

Data preprocessing: The min-max normalization [35] is applied to preprocess datasets, aiming to avoid the di GLYPH&lt;11&gt; erence of dimensional metrics.

Machine configuration: MATLAB (R2019b) on Windows Server 2019 Datacenter with Intel 2 Xeon 2 Gold 5220R CPU at 2.20 GHz (2 processors) and 256 GB RAM.

Evaluation metric: The popular adjusted rand index (ARI) [43], adjusted mutual information (AMI) [43], normalized mutual information (NMI) [44], Fowlkes-Mallows index (FMI) [45], accuracy (ACC), and F -score [46] are used for the clustering performance evaluation. Besides, the F 1-score [48] is applied to measure the robustness of center detection.

## B. Results on Synthetic Datasets

Fig. 9 presents the clustering results of di GLYPH&lt;11&gt; erent algorithms. Here, ' F ' denotes detected cluster centers, ' × ' marks identified noise points, and distinct colors correspond to di GLYPH&lt;11&gt; erent clusters.

As illustrated in Fig. 9, PeakPad achieves near-perfect clustering across all tested datasets. DBSCAN performs well in reconstructing complex shapes but tends to misclassify many border points as noise in the Agg , Compound , and S3 datasets. Although DPC correctly clusters Agg , Flame , and S3 , it fails on Jain , Threecircles , Compound , and T48K datasets due to inaccurate cluster center identification. MSC produces satisfactory results on Flame and S3 , but su GLYPH&lt;11&gt; ers from oversegmentation on Threecircles , Compound , and T48K , and incorrect center detection on Jain . K -means fails to handle all complex-shaped datasets.

Table II summarizes quantitative scores of all algorithms, with the best results highlighted. PeakPad consistently achieves top performance across nearly all synthetic datasets. sparse subspace clustering (SSC) outperforms K -means on complex shape recognition tasks. Notably, improved DPC variants-SSSP-DPC, SNN-DPC, TC, and

2 Registered trademark.

TABLE II COMPARISON ON SYNTHETIC DATASETS (METRICS: AMI/ARI j NMI/F-SCORE j FMI/ACC)

DBSCAN-DPC-demonstrate enhanced accuracy compared to the original DPC. Furthermore, GB-DP, DBSCAN-DPC, and UP-DPC exhibit superior scalability, e GLYPH&lt;14&gt; ciently processing large datasets such as Birchrg 1 and TB 10 w with 100 000 points each. In contrast, DPC and several of its improved versions (SSSP-DPC, SNN-DPC, and TC) fail to execute under the current computational environment due to excessive memory demands caused by their algorithmic complexity.

Overall, the test results validate that PeakPad delivers robust and accurate recognition of complex cluster shapes, demonstrating superior adaptability and computational e GLYPH&lt;14&gt; ciency compared to competing methods.

## C. Results on Real-World Datasets

To further evaluate the clustering performance of PeakPad on large-scale and high-dimensional datasets, we conducted benchmark tests on ten widely used real-world datasets. These include nine UCI datasets ( Iris , Wine , Segment , Drivepoints , Breastcancer , Waveform , Movementlibras , Parkin , and Vote ), three large-scale machine learning datasets ( YTF , MNIST preprocessed by [7]), and the well-known OlivettiFaces dataset [42] preprocessed by [9]. Table I summarizes the key characteristics of these datasets, encompassing a variety of data dimensions and cluster complexities.

The quantitative results presented in Table III demonstrate that PeakPad consistently delivers superior or competitive performance across all datasets, with the best results highlighted. Furthermore, Table IV lists the parameter settings for all comparison algorithms (except for the parameter-free TC algorithm [50]), ensuring a fair evaluation.

Fig. 10 visualizes the clustering results on selected complex, high-dimensional datasets using t-SNE [49]. It can be observed that both DBSCAN-DPC and PeakPad achieve clustering structures closely aligned with the true labels, particularly on challenging datasets like MNIST . This highlights PeakPad's ability to e GLYPH&lt;11&gt; ectively capture intrinsic data structures even in complex feature spaces.

Overall, these extensive tests validate PeakPad as a robust and e GLYPH&lt;11&gt; ective clustering method, capable of handling diverse and high-dimensional real-world data, thus demonstrating its potential for practical applications.

## D. Robustness of Center Detection

F 1-score [48] [as in (26)] is used to quantify the center detection robustness of PeakPad, where true positives (TPs) refers to the correctly identified cluster centers that match the ground-truth centers (i.e., the highest density points in each true cluster), false positives (FPs) refers to the incorrect or extra cluster centers detected, and false negatives (FNs) refers to the ground-truth centers that were missed by the algorithm.

To verify the superiority of PeakPad, for each test, each DPC-based algorithm ( GLYPH&lt;13&gt; -based center detection) directly selects its true number of centers with the top largest GLYPH&lt;13&gt; (i.e., GLYPH&lt;26&gt; GLYPH&lt;1&gt; GLYPH&lt;14&gt; ) values to obtain the F 1-score, while PeakPad ( GLYPH&lt;18&gt; -based center detection) selects the true number of centers with the

TABLE III COMPARISON ON REAL-WORLD DATASETS (METRICS: AMI/ARI j NMI/F-SCORE j FMI/ACC)

TABLE IV PARAMETER SETTINGS OF DIFFERENT ALGORITHMS ON THE TESTED DATASETS

top largest GLYPH&lt;18&gt; values. The test results are reported in Table V, with the best results highlighted. As shown, PeakPad gets the highest scores on all datasets (except for Segment ), verifying the e GLYPH&lt;11&gt; ectiveness of PeakPad in center detection

<!-- formula-not-decoded -->

Fig. 11 shows the clustering results of DPC and PeakPad by selecting forty centers on the OlivettiFaces [42] dataset, where cluster centers are marked in white circles. As Fig. 11 shows, in terms of clustering e GLYPH&lt;11&gt; ect, PeakPad is superior; and in terms of center detection, DPC incorrectly identified nine centers (marked by white boxes) with F 1 = 0.78, while PeakPad only misidentified four centers with F 1 = 0.90.

Fig. 10. t-SNE-based visualization comparison of di GLYPH&lt;11&gt; erent algorithms on real-world datasets.

<!-- image -->

Fig. 11. Clustering results of DPC and PeakPad on the OlivettiFaces dataset.

<!-- image -->

To further validate the e GLYPH&lt;11&gt; ectiveness of PeakPad, we conducted tests on 160 synthetic datasets with varying geometric structures and dimensions, including 80 Gaussian sphere datasets and 80 ellipsoidal cluster datasets [54]. These were designed to rigorously evaluate the robustness of the proposed

GLYPH&lt;18&gt; -based center detection mechanism. All algorithms were tested with fixed, well-tuned parameters. As shown in Fig. 12, PeakPad consistently outperforms other methods in both center detection and overall clustering accuracy, achieving average F 1-scores of 0.9588 and 0.8788 on Gaussian and ellipsoidal

TABLE V

## COMPARISON OF CENTER DETECTION PERFORMANCE ( F 1-SCORE)

Fig. 12. Comparative performance of DPC-based algorithms and PeakPad on 160 synthetic datasets (80 Gaussian spheres and 80 ellipsoids).

<!-- image -->

Fig. 13. (a) Runtime comparison of di GLYPH&lt;11&gt; erent algorithms and (b) runtime comparison between K -means and ours on ten di GLYPH&lt;11&gt; erent size sampling datasets of the Birchrg1 dataset.

<!-- image -->

datasets, respectively. It also ranks highest across all seven evaluation metrics (e.g., AMI, ARI, and NMI), demonstrating the geometric adaptability of the proposed d GLYPH&lt;18&gt; -based center detection mechanism.

As verified, PeakPad owns a more robust center detection performance than other DPC-based algorithms, which is also sound evidence of the e GLYPH&lt;11&gt; ectiveness and robustness of our new center assumption.

## E. Speed of PeakPad

Execution speed is also an important criterion for evaluating an algorithm's performance in large-scale data clustering tasks. Fig. 13(a) presents the runtime comparison of all comparison algorithms on di GLYPH&lt;11&gt; erent datasets. As shown, PeakPad and K -means outperform other algorithms, but is much faster than DPC and SNN-DPC (an improved version of DPC), especially for large-scale datasets, such as YTF and USPS .

To further verify the fast speed of PeakPad, we conducted speed comparisons between PeakPad and K -means using the Birchrg1 dataset which consisted of 100 000 points, as in Fig. 13(b). As shown, PeakPad is faster than K -means. When handling 100 000 data points, PeakPad takes only 1.5 s,

Fig. 14. k -AMI plot of the PeakPad algorithm on di GLYPH&lt;11&gt; erent datasets with k 2 [0 ; 2 d ( n ) 1 = 2 e ].

<!-- image -->

which is 1 s faster than K -means. Moreover, PeakPad also outperforms K -means in terms of accuracy (see Table II).

As verified, PeakPad with fast speed is promising for largescale data clustering.

## F. Parameter Insensitivity

The main parameter k (i.e., the number of neighbors) in PeakPad is set k = d ( n ) 1 = 2 e as default. Parameter k is needed in our kNN-based density estimation, density peak detection, and peak graph building. Thus, the parameter insensitivity of k deserves a discussion.

Fig. 14 presents the k -AMI plot of several tested datasets, with k 2 [0 ; 2 d ( n ) 1 = 2 e ] and a given correct cluster number as inputs. As Fig. 14 shows, within the range of [0 ; 2 d ( n ) 1 = 2 e ], PeakPad obtains a stable optimal performance over a wide range around k = d ( n ) 1 = 2 e , which verifies the e GLYPH&lt;11&gt; ectiveness of k = d ( n ) 1 = 2 e setting and the insensitivity of PeakPad to parameter k .

Except for parameter k , PeakPad also needs nc (cluster number) and kb (border point detection parameter). nc is selected by observing our decision graph, just like in DPC, and kb is a fixed setting as in (2).

## V. CONCLUSION

The proposed Peak-Padding Clustering (PeakPad) algorithm addresses the challenge of clustering complex-shaped clusters by decomposing them into high-association single-peak clusters. This transformation simplifies the clustering task from assigning data points to clusters to associating density peaks. By introducing the concept of DC distance, PeakPad converts arbitrary-shaped density surface paths into standardized DC surface paths, enabling more accurate evaluation of associations among single-peak clusters.

PeakPad identifies cluster centers as density peaks with relatively high minimum padding costs in the peak graph and assigns noncenter peaks (representing single-peak clusters) to their corresponding complex-shaped clusters. This approach yields accurate complex shapes clustering and robust center detection. Furthermore, due to its reliance on kNN-based distances, PeakPad is e GLYPH&lt;14&gt; cient and scalable to large datasets. Its e GLYPH&lt;11&gt; ectiveness has been validated on both synthetic and realworld datasets, demonstrating strong performance in clustering and center detection.

However, PeakPad's clustering performance depends heavily on the quality of single-peak cluster identification, which in turn relies on accurate density estimation. Currently, a simple kNN-based estimator is used for e GLYPH&lt;14&gt; ciency, but there is room for improvement, especially in high-dimensional or noisy data scenarios [22], [51], [52]. Future work could incorporate nonlinear dimensionality reduction and feature selection techniques to enhance the accuracy and robustness of density estimation. Adaptive imputation strategies, such as evolving fuzzy systems, may also help enable real-time, uncertaintyaware clustering.

Although PeakPad shows strong performance in center detection, it still requires manual selection of cluster centers. As automatic center detection is critical in many real-world applications, future enhancements will focus on integrating Bayesian nonparametric models [53] and dynamic update mechanisms to enable fully automatic cluster center identification.
