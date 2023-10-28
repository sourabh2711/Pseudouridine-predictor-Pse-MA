 # Pseudouridine-predictor-Pse-MA
RNA protein Interactions (RPIs) play an important role in biological systems. Recently, we have enumerated the RPIs at residue level and have elucidated the minimum structural unit (MSU) in these
interactions to be a stretch of five residues (Nucleotides/amino acids).
Pseudouridine is the most frequent modification in RNA. The conversion of uridine to pseudouridine involves interactions between pseudourdine synthase and RNA. The existing models to predict the pseudouridine sites in a given RNA sequence mainly depend on user defined features such as mono and dinucleotide composition/propensities of RNA
sequences. Predicting pseudouridine sites is a non-linear classification
problem with limited data points. Deep Learning models are efficient
discriminators when the data set size is reasonably large and fails when
there is paucity in data (< 1000 samples). To mitigate this problem,
we propose a Support Vector Machine (SVM) Kernel based on utility
theory from Economics, and using data driven parameters (i.e. MSU) as
features. For this purpose, we have used position-specific pentanucleotide
composition/propensity (PSPC/PSPP) as features. SVMs are known to
work well in small data regime and kernels in SVM are designed to classify non-linear data. The proposed model outperforms the existing state
of the art models significantly (10% − 15% on average).
