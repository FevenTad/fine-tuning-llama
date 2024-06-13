# !pip install wandb
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes

from huggingface_hub import notebook_login
notebook_login()

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# alpaca prompt structure
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# call the model from hugging face
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "FevenTad/promt_eng_model_10E",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model) # Enable 2x faster inference

inputs = tokenizer(
[
    alpaca_prompt.format(
        "SSiSirirrrrara a g gogo o y yoyouou u t toto o Bo BaBarartrththohololmlmemewew w m mymy y P PaPagagege,e,,AAnAndnd d s seseeee e h hihimim m d drdreresestst t i inin n a alallll l s su suiuititeteses es l lilikikeke e ae a a La LaLadadidieie:e::TThThahatat t dt dodononene,e, e, c coconondonduducuctct t ht hihimhim m tm toto to t th thehe e de drdrudrununknkakarardardsds s c ch chahamhambmbeberer,r,,A,AnAndAnd d cd cacalallall l hl hihimhim him M MaMadadaadamam,m, , d do do o ho hihimhim him ohim obobebeibeisisasanancncece:e:e:TTeTelellll ll h hi himim im f frfroromom m mm meme me ( (a(asas s hs hehe he w wiwilillll ll wll wiwinwin n mn mymy my l lo lououeoue)e))HHeHe e be bebeabearareare e he hihimhimshimseselselflfefe e we wiwitwithth th h ho hononoonouourourarabrablblele e ae ace actctictioionon,on,,S,SuSucuchuch h a as as s hs hes he e he hae hatathath h oh obobsobseserseruru'u'd'd d id inin in n nonobobloblele le L LaLadLadidiediesesesVVnVntntoto to tto ththetheieirir ir L Lo Lorordrdsds,ds, , b, byby y ty ththethemthem m am acaccccocomcompmplplilislishshehedhed,d,,S,Su,Sucuchuch uch d du dututitieie ie t to to o to ththethe the d dr drurunrunknkankarardard ard l le letet t ht hit himim im dim dodo:do", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
tokenizer.batch_decode(outputs)