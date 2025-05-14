import os
import numpy as np
from PIL import Image
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ========== 1. Fuzzy Logic Setup ==========
texture_similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'texture_similarity')
color_similarity = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'color_similarity')
fusion_weight = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'fusion_weight')  # TCP weight

texture_similarity.automf(3)
color_similarity.automf(3)

fusion_weight['low'] = fuzz.trimf(fusion_weight.universe, [0, 0, 0.5])
fusion_weight['medium'] = fuzz.trimf(fusion_weight.universe, [0.25, 0.5, 0.75])
fusion_weight['high'] = fuzz.trimf(fusion_weight.universe, [0.5, 1.0, 1.0])

rule1 = ctrl.Rule(texture_similarity['poor'] & color_similarity['poor'], fusion_weight['high'])
rule2 = ctrl.Rule(texture_similarity['average'] & color_similarity['average'], fusion_weight['medium'])
rule3 = ctrl.Rule(texture_similarity['good'] & color_similarity['good'], fusion_weight['low'])

fusion_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fusion_simulator = ctrl.ControlSystemSimulation(fusion_ctrl)

# ========== 2. Image Similarity ==========
def compute_texture_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return 1 - np.mean(np.abs(gray1.astype(float) - gray2.astype(float)) / 255.0)

def compute_color_similarity(img1, img2):
    return 1 - np.mean(np.abs(img1.astype(float) - img2.astype(float)) / 255.0)

# ========== 3. Image Fusion ==========
def fuzzy_fusion(tcp_img_path, ai_img_path, output_path='fused_output.jpg'):
    tcp = cv2.imread(tcp_img_path)
    ai = cv2.imread(ai_img_path)

    tcp = cv2.resize(tcp, (256, 256))
    ai = cv2.resize(ai, (256, 256))

    tex_sim = compute_texture_similarity(tcp, ai)
    col_sim = compute_color_similarity(tcp, ai)

    fusion_simulator.input['texture_similarity'] = tex_sim
    fusion_simulator.input['color_similarity'] = col_sim
    fusion_simulator.compute()

    tcp_weight = fusion_simulator.output['fusion_weight']
    ai_weight = 1 - tcp_weight

    fused = cv2.addWeighted(tcp, tcp_weight, ai, ai_weight, 0)
    cv2.imwrite(output_path, fused)
    return output_path, tcp_weight, ai_weight

# ========== 4. Semantic Analysis using BLIP ==========
def semantic_analysis_blip(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# ========== 5. Main Execution ==========
if __name__ == "__main__":
    tcp_path = "input_images/tcp.jpg"
    ai_path = "input_images/ai.jpg"

    fused_output_path, tcp_w, ai_w = fuzzy_fusion(tcp_path, ai_path)

    caption = semantic_analysis_blip(fused_output_path)

    print("\n🖼️ Fused image saved to:", fused_output_path)
    print(f"🎨 Fusion Weights → TCP: {tcp_w:.2f}, AI: {ai_w:.2f}")
    print("🧠 Semantic Caption:", caption)

    fused = cv2.imread(fused_output_path)
    fused_rgb = cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)

    plt.imshow(fused_rgb)
    plt.title("Fused Image\n" + caption)
    plt.axis('off')
    plt.show()
