# !pip install wandb
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
import wandb

wandb.login()

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = torch.float16 # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,

    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    use_gradient_checkpointing = True, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from huggingface_hub import notebook_login
notebook_login()

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("FevenTad/dataset_2", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        # max_steps = None,
        num_train_epochs=10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to="wandb",
        output_dir="./results/",
    ),
)

trainer_stats = trainer.train()

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    alpaca_prompt.format(
        "ccacatat t w whwhoho o c cacancannnnonotot t tt totololelereraratateate e T THTHIHISIS S P PAPARARtRtiticicucululalarar r F FaFanancncycy y F Fe Feaeasastst t St SaSalalmlmomonon n P Pr Prorododuducuctct.t. . I ItIt t it isis s o ononlonlyly y ty ththehe e Se ShShrhrerededdddededed ed o on onene e te ththathatat at g gegetetsts s hs heherher,r, , b bubutut t gt gegetget t it itt it t dt dodoeoeses.s.<.<b<brbr r /r />/>E>EvEvevereryerytytitimtimeme e se shshehe he e eaeateatsts ts i itit,t, , s, shsheshe e te the thrhrohrowowsws s is itit it u upup p a alallall l o ov oveverver r er eveveververyverywywhwhewherereere.e. . M. MyMy y sy shshoshoeoesoes,s, , t, ththethe e we whwhiwhititeite e re rurugug g a an andnd d g gr graratratetefefufulullullyly,ly, , t, th, thehe he the titiltilele.le.<.<b.<brbr br / />/></><b<br<br r /r />r />S>ShSheShe e ae atateate ate t ththithisis is f foforor r sr seseveveevereraeralal al y yeyeaearearsrs s ws wiwitithith h o oc occccacascasisioiononaonalal al ual upupsupsesetsetsts,ts, , b, bubutbut t nt nonownow w i it it t it ist is s Es EVEVEVERERYRY Y t ti timimeme.me. . N. NoNo o mo momormorere re f fo foror or t th thihisis is fis fafamamimililyly.ly. . I. I I r rerearealalializizeze e te the thae thatat at tat ththithisthis s os ococcccuccurursrs rs o on on  on a an an  an i in indndindivivivididuduadualal al bal babasasiasisis is Uis USUSUSUAUALALLLLYLY.Y.<.<b.<br.<br r /r />r /><r /><b<br<br <br / /> />B>BuButut,ut, , I, I I aI alalsalsoso o ho hahavaveve ve o on one one e ye yoyouounungngegerger r kr kikitittittyty y wy whwhowho o io isis is ais atat at tat that thehe he \"he \"\"\"\"o\"ococcoccacascasicasioionionanalal al ual upal upspsesetset\"set\"\"", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
tokenizer.batch_decode(outputs)

# model.save_pretrained("test_version") # Local saving
# tokenizer.save_pretrained("test_version")

# model.push_to_hub("promt_eng_model_10E")
# tokenizer.push_to_hub("promt_eng_model_10E")