# ============================================================
#  CodSoft Artificial Intelligence Internship
#  Task 03: Image Captioning AI
#  Author : Jaspreet Singh
#  GitHub : https://github.com/Jaspreet-Singh7/CODSOFT
#
#  Approach:
#  - Computer Vision: Analyze image colors, brightness,
#    composition, shapes using OpenCV & PIL
#  - NLP: Generate natural language captions using
#    rule-based template system with detected features
#  - Simulates VGG/ResNet feature extraction + RNN caption
#    generation pipeline conceptually
# ============================================================

import cv2
import numpy as np
import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import datetime

OUTPUT_DIR = "image_captioning_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── FEATURE EXTRACTOR ────────────────────────────────────────
class ImageFeatureExtractor:
    """
    Extracts visual features from images.
    Simulates what VGG/ResNet does in deep learning pipeline.
    Analyzes: colors, brightness, composition, dominant objects.
    """

    def __init__(self):
        self.color_names = {
            "red":    ([0,0,150],   [100,100,255]),
            "green":  ([0,100,0],   [100,255,100]),
            "blue":   ([100,0,0],   [255,100,100]),
            "yellow": ([0,150,150], [100,255,255]),
            "white":  ([200,200,200],[255,255,255]),
            "black":  ([0,0,0],     [60,60,60]),
            "orange": ([0,100,150], [100,200,255]),
            "purple": ([100,0,100], [200,100,200]),
        }

    def extract(self, image_path):
        """Extract all visual features from image."""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            # Try with PIL
            try:
                pil_img = Image.open(image_path).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_bgr.shape[:2]

        features = {
            "size":         (w, h),
            "aspect_ratio": round(w/h, 2),
            "brightness":   self._get_brightness(img_bgr),
            "dominant_colors": self._get_dominant_colors(img_rgb),
            "color_mood":   self._get_color_mood(img_rgb),
            "composition":  self._get_composition(img_bgr),
            "edge_density": self._get_edge_density(img_bgr),
            "texture":      self._get_texture(img_bgr),
            "scene_type":   self._classify_scene(img_bgr),
        }
        return features

    def _get_brightness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        if mean > 180:   return "very bright"
        elif mean > 130: return "bright"
        elif mean > 80:  return "moderate"
        elif mean > 40:  return "dark"
        else:            return "very dark"

    def _get_dominant_colors(self, img_rgb, n_colors=3):
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10,
                                         cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        sorted_colors = centers[np.argsort(-counts)]

        color_names = []
        for color in sorted_colors:
            r, g, b = color
            name = self._name_color(r, g, b)
            color_names.append(name)
        return color_names

    def _name_color(self, r, g, b):
        colors = {
            "red":    (200, 50,  50),
            "green":  (50,  180, 50),
            "blue":   (50,  50,  200),
            "yellow": (220, 220, 50),
            "orange": (220, 130, 50),
            "purple": (150, 50,  180),
            "cyan":   (50,  200, 200),
            "white":  (230, 230, 230),
            "black":  (30,  30,  30),
            "gray":   (128, 128, 128),
            "brown":  (130, 80,  40),
            "pink":   (220, 120, 150),
        }
        min_dist = float('inf')
        closest = "gray"
        for name, (cr, cg, cb) in colors.items():
            dist = ((r-cr)**2 + (g-cg)**2 + (b-cb)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    def _get_color_mood(self, img_rgb):
        mean_r = np.mean(img_rgb[:,:,0])
        mean_g = np.mean(img_rgb[:,:,1])
        mean_b = np.mean(img_rgb[:,:,2])

        saturation = np.std(img_rgb)

        if saturation < 30:      return "monochromatic"
        elif mean_r > mean_b + 30: return "warm"
        elif mean_b > mean_r + 30: return "cool"
        elif mean_g > 150:         return "natural"
        else:                      return "neutral"

    def _get_composition(self, img):
        h, w = img.shape[:2]
        regions = {
            "top":    img[:h//3, :],
            "middle": img[h//3:2*h//3, :],
            "bottom": img[2*h//3:, :],
            "left":   img[:, :w//3],
            "center": img[:, w//3:2*w//3],
            "right":  img[:, 2*w//3:],
        }
        brightness_map = {k: np.mean(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY))
                          for k, v in regions.items()}
        brightest = max(brightness_map, key=brightness_map.get)
        return brightest

    def _get_edge_density(self, img):
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = np.sum(edges > 0) / edges.size
        if density > 0.15:   return "high detail"
        elif density > 0.06: return "medium detail"
        else:                return "low detail"

    def _get_texture(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        std  = np.std(gray)
        if std > 60:   return "complex texture"
        elif std > 30: return "moderate texture"
        else:          return "smooth"

    def _classify_scene(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean_bright = np.mean(gray)
        mean_g = np.mean(img_rgb[:,:,1])
        mean_b = np.mean(img_rgb[:,:,2])
        mean_r = np.mean(img_rgb[:,:,0])
        edge_d = np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size

        if mean_g > mean_r + 20 and mean_g > mean_b: return "nature/outdoor"
        elif mean_b > mean_r + 20:                   return "sky/water"
        elif edge_d > 0.12 and mean_bright < 150:    return "urban/city"
        elif mean_bright > 180 and edge_d < 0.05:    return "minimalist"
        elif edge_d > 0.1:                           return "complex/detailed"
        else:                                         return "general"


# ── CAPTION GENERATOR ────────────────────────────────────────
class CaptionGenerator:
    """
    Generates natural language captions from image features.
    Simulates RNN/Transformer decoder in deep learning pipeline.
    Uses template-based NLG (Natural Language Generation).
    """

    def __init__(self):
        self.templates = {
            "nature/outdoor": [
                "A {brightness} outdoor scene featuring {color1} and {color2} tones with {texture}.",
                "A {mood} natural landscape with predominantly {color1} hues and {detail} composition.",
                "An outdoor photograph showcasing {color1} and {color2} colors in a {mood} setting.",
                "A {brightness} nature scene with {texture} and {color1} dominant colors.",
            ],
            "sky/water": [
                "A {brightness} scene featuring {color1} and {color2} tones, suggesting sky or water.",
                "A {mood} aerial or waterscape with {color1} hues and {texture}.",
                "A serene {brightness} scene with {color1} and {color2} tones dominating the frame.",
            ],
            "urban/city": [
                "An urban scene with {detail}, featuring {color1} and {color2} color palette.",
                "A city photograph with {texture} and {brightness} lighting conditions.",
                "A {mood} urban setting with {color1} tones and {detail} architectural elements.",
            ],
            "minimalist": [
                "A minimalist composition featuring {color1} and {color2} with clean, simple lines.",
                "A {brightness} minimalist image with {color1} dominant colors and smooth texture.",
                "A simple, {brightness} scene with {color1} and {color2} in a clean composition.",
            ],
            "general": [
                "An image featuring {color1} and {color2} tones with {texture} in a {mood} palette.",
                "A {brightness} photograph with {detail} and predominantly {color1} colors.",
                "A {mood} scene featuring {color1} and {color2} with {texture} composition.",
                "An image with {color1} dominant colors, {brightness} lighting and {texture}.",
            ],
        }

        self.detail_phrases = {
            "high detail":    "intricate details",
            "medium detail":  "moderate detail",
            "low detail":     "clean composition",
        }

    def generate(self, features, num_captions=3):
        """Generate multiple captions from extracted features."""
        if not features:
            return ["Unable to analyze image."]

        scene = features.get("scene_type", "general")
        templates = self.templates.get(scene, self.templates["general"])

        colors = features.get("dominant_colors", ["gray", "white", "black"])
        color1 = colors[0] if len(colors) > 0 else "neutral"
        color2 = colors[1] if len(colors) > 1 else "white"

        detail = self.detail_phrases.get(
            features.get("edge_density", "medium detail"), "moderate detail")

        fill = {
            "brightness": features.get("brightness", "moderate"),
            "color1":     color1,
            "color2":     color2,
            "mood":       features.get("color_mood", "neutral"),
            "texture":    features.get("texture", "varied texture"),
            "detail":     detail,
            "composition":features.get("composition", "center"),
        }

        captions = []
        used = set()

        # Use each template once
        shuffled = random.sample(templates, min(num_captions, len(templates)))
        for template in shuffled:
            caption = template.format(**fill)
            # Capitalize first letter
            caption = caption[0].upper() + caption[1:]
            if caption not in used:
                captions.append(caption)
                used.add(caption)

        # Add a detailed caption
        detailed = (
            f"This {features.get('brightness', 'moderate')} image ({features['size'][0]}x{features['size'][1]}px) "
            f"displays {color1} and {color2} as dominant colors with a {features.get('color_mood','neutral')} "
            f"color mood. The scene appears to be {scene} with {detail} "
            f"and {features.get('texture', 'varied')} throughout."
        )
        if detailed not in used:
            captions.append(detailed)

        return captions[:num_captions]


# ── IMAGE CAPTIONING SYSTEM ───────────────────────────────────
class ImageCaptioningSystem:
    """Main image captioning pipeline combining CV + NLP."""

    def __init__(self):
        self.extractor = ImageFeatureExtractor()
        self.generator = CaptionGenerator()
        print("✅ Image Captioning System initialized")
        print("   Feature Extractor: OpenCV + Color Analysis")
        print("   Caption Generator: Template-based NLG")

    def caption_image(self, image_path, num_captions=3):
        """Full pipeline: image → features → captions."""
        print(f"\n🖼️  Processing: {os.path.basename(image_path)}")

        # Extract features
        features = self.extractor.extract(image_path)
        if not features:
            return None, ["Could not process image."]

        print(f"   Size: {features['size'][0]}x{features['size'][1]}")
        print(f"   Brightness: {features['brightness']}")
        print(f"   Dominant Colors: {features['dominant_colors'][:2]}")
        print(f"   Color Mood: {features['color_mood']}")
        print(f"   Scene Type: {features['scene_type']}")
        print(f"   Edge Density: {features['edge_density']}")

        # Generate captions
        captions = self.generator.generate(features, num_captions)

        return features, captions

    def create_captioned_image(self, image_path, captions, features):
        """Create output image with captions overlaid."""
        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # Add dark caption bar at bottom
        caption_height = 35 + len(captions) * 30
        canvas = np.zeros((h + caption_height, w, 3), dtype=np.uint8)
        canvas[:h, :w] = img
        canvas[h:, :] = (30, 30, 30)

        # Title bar
        cv2.rectangle(canvas, (0, h), (w, h+32), (50, 50, 50), -1)
        cv2.putText(canvas, "AI Generated Captions | CodSoft Task 03 | Jaspreet Singh",
                   (5, h+22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Captions
        for i, caption in enumerate(captions):
            y = h + 50 + i * 28
            # Truncate if too long
            max_chars = int(w / 8)
            display = caption[:max_chars] + "..." if len(caption) > max_chars else caption
            cv2.putText(canvas, f"{i+1}. {display}",
                       (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1)

        return canvas


# ── SYNTHETIC TEST IMAGES ─────────────────────────────────────
def create_test_images():
    """Create synthetic test images for demonstration."""
    images = []

    # Image 1: Nature scene (green)
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img1[:150, :] = [34, 139, 34]    # Green sky
    img1[150:, :] = [0, 100, 0]      # Dark green ground
    # Sun
    cv2.circle(img1, (80, 60), 40, (0, 220, 255), -1)
    # Trees
    for x in [100, 200, 300]:
        cv2.rectangle(img1, (x-10, 130), (x+10, 200), (60, 30, 10), -1)
        cv2.circle(img1, (x, 110), 40, (0, 160, 0), -1)
    path = f"{OUTPUT_DIR}/test_image_nature.jpg"
    cv2.imwrite(path, img1)
    images.append(("Nature Scene", path))

    # Image 2: Sky/water scene (blue)
    img2 = np.zeros((300, 400, 3), dtype=np.uint8)
    for y in range(300):
        blue = int(100 + y * 0.5)
        img2[y, :] = [blue, int(blue*0.6), 30]
    # Clouds
    for cx, cy in [(100, 50), (280, 80), (200, 30)]:
        cv2.ellipse(img2, (cx, cy), (50, 25), 0, 0, 360, (255, 255, 255), -1)
    path = f"{OUTPUT_DIR}/test_image_sky.jpg"
    cv2.imwrite(path, img2)
    images.append(("Sky Scene", path))

    # Image 3: Urban scene (gray/complex)
    img3 = np.ones((300, 400, 3), dtype=np.uint8) * 180
    # Buildings
    buildings = [(20,80,80,220),(120,60,80,240),(220,40,80,260),(320,70,70,230)]
    for (x, top, w, h) in buildings:
        cv2.rectangle(img3, (x, top), (x+w, 300), (100+random.randint(0,50),)*3, -1)
        # Windows
        for wy in range(top+10, 280, 25):
            for wx in range(x+5, x+w-10, 15):
                cv2.rectangle(img3, (wx, wy), (wx+8, wy+15),
                             (200, 220, 255) if random.random()>0.3 else (50,50,50), -1)
    # Road
    cv2.rectangle(img3, (0, 260), (400, 300), (60, 60, 60), -1)
    cv2.line(img3, (200, 260), (200, 300), (255, 255, 255), 3)
    path = f"{OUTPUT_DIR}/test_image_urban.jpg"
    cv2.imwrite(path, img3)
    images.append(("Urban Scene", path))

    # Image 4: Minimalist (white/simple)
    img4 = np.ones((300, 400, 3), dtype=np.uint8) * 245
    cv2.circle(img4, (200, 150), 80, (220, 50, 50), -1)
    cv2.rectangle(img4, (100, 240), (300, 280), (50, 50, 220), -1)
    path = f"{OUTPUT_DIR}/test_image_minimalist.jpg"
    cv2.imwrite(path, img4)
    images.append(("Minimalist Scene", path))

    return images


# ── DEMO RUN ──────────────────────────────────────────────────
def run_demo():
    """Full demo of image captioning system."""
    print("\n" + "=" * 60)
    print("  CodSoft AI Task 03: Image Captioning")
    print("  CV + NLP Pipeline | Author: Jaspreet Singh")
    print("=" * 60)

    system = ImageCaptioningSystem()
    all_passed = True

    print("\n📸 Creating synthetic test images...")
    test_images = create_test_images()
    print(f"✅ Created {len(test_images)} test images in ./{OUTPUT_DIR}/")

    print("\n" + "-" * 60)
    print("🤖 GENERATING CAPTIONS FOR ALL TEST IMAGES")
    print("-" * 60)

    results = []
    for scene_name, img_path in test_images:
        print(f"\n{'='*50}")
        print(f"📷 Scene: {scene_name}")

        features, captions = system.caption_image(img_path, num_captions=3)

        if captions:
            print(f"\n📝 Generated Captions:")
            for i, cap in enumerate(captions, 1):
                print(f"   {i}. {cap}")

            # Save captioned image
            if features:
                output_img = system.create_captioned_image(img_path, captions[:2], features)
                out_path = img_path.replace(".jpg", "_captioned.jpg")
                cv2.imwrite(out_path, output_img)
                print(f"   💾 Saved: {out_path}")

            results.append((scene_name, captions))
            passed = len(captions) >= 2 and all(len(c) > 20 for c in captions)
            print(f"   {'✅ PASSED' if passed else '❌ FAILED'}")
            if not passed: all_passed = False
        else:
            print("   ❌ Caption generation failed!")
            all_passed = False

    print("\n\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"  Images processed : {len(test_images)}")
    print(f"  Captions generated: {sum(len(r[1]) for r in results)}")
    print(f"  Output directory  : ./{OUTPUT_DIR}/")

    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
    print(f"  Output files      : {len(files)} images")
    print(f"\n  Pipeline Used:")
    print(f"  📷 Input Image")
    print(f"  ↓ OpenCV Feature Extraction (colors, brightness, edges)")
    print(f"  ↓ Scene Classification (nature/urban/sky/minimalist)")
    print(f"  ↓ NLG Caption Generation (template + features)")
    print(f"  ✅ Natural Language Caption Output")
    print(f"\n{'✅ All tests passed!' if all_passed else '⚠️  Some tests need review'}")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    run_demo()
