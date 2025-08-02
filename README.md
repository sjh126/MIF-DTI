# MIF-DTI
MIF-DTI: a multimodal information fusion method for drug-target interaction prediction

# Dependencies:
python == 3.8
torch == 2.4.1
torch_geometric
pyg_lib
torch_scatter
joblib
numpy
prefetch_generator
scikit-learn
tqdm
pandas
fair-esm
Bio
rdkit

# Resources:
README.md: this file.
requirements.txt:  dependencies.
DataSets: DrugBank.txt, Davis.txt, BioSNAP.txt, BD2D.txt.
RunModel.py: train and test the model.
main.py: main process.
model.py: MIF and MIF-B model architecture.

#Run:
python -u main.py [dataset] -m [model]
