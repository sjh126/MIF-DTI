from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

protein = ESMProtein(sequence= "MMMMMMMMMMMMM")
                     
client = ESMC.from_pretrained("esmc_600m").to("cuda") # or "cpu"
protein_tensor = client.encode(protein)
logits_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
print(logits_output.logits, logits_output.embeddings)
print(logits_output.embeddings.shape) #[1,seq_len+2,1152]