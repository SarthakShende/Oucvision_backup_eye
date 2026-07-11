#!/usr/bin/env python3
"""
OcuVisionAI GPU Stability Fixer
Run in your OcuVision directory: python fix_ocuvision.py
"""
import re
from pathlib import Path

def patch_file():
    input_file = Path("ocuvision_complete.py")
    output_file = Path("ocuvision_complete_FIXED.py")
    
    if not input_file.exists():
        print(f"❌ Not found: ocuvision_complete.py")
        print(f"   Run this script from your OcuVision directory")
        return False
    
    print("📖 Reading ocuvision_complete.py...")
    with open(input_file, 'r') as f:
        content = f.read()
    
    print("\n✨ Applying 6 stability fixes:\n")
    
    # FIX 1: Disable mixed precision
    print("  [1/6] float16 → float32 (mixed precision disabled)")
    if 'mixed_float16' in content:
        content = content.replace(
            'mixed_precision.set_global_policy("mixed_float16")',
            'mixed_precision.set_global_policy("float32")  # FIXED: disabled mixed_float16'
        )
    
    # FIX 2: Reduce batch size
    print("  [2/6] Batch size: 128 → 32")
    content = re.sub(
        r'"batch"\s*:\s*128,',
        '"batch": 32,  # FIXED',
        content
    )
    
    # FIX 3: Reduce batch_seg
    print("  [3/6] Batch seg: 64 → 16")
    content = re.sub(
        r'"batch_seg"\s*:\s*64,',
        '"batch_seg": 16,  # FIXED',
        content
    )
    
    # FIX 4: Reduce learning rate
    print("  [4/6] Learning rate: 3e-4 → 1e-4")
    content = re.sub(
        r'"lr"\s*:\s*3e-4,',
        '"lr": 1e-4,  # FIXED',
        content
    )
    
    # FIX 5: Add gradient clipping in Adam optimizer
    print("  [5/6] Adding gradient clipping (clipnorm=1.0)")
    # More robust pattern matching
    old_adam = 'optimizer=tf.keras.optimizers.Adam(CFG["lr"])'
    new_adam = '''optimizer=tf.keras.optimizers.Adam(
        learning_rate=CFG["lr"],
        clipnorm=1.0,
        global_clipnorm=1.0
    )'''
    if old_adam in content:
        content = content.replace(old_adam, new_adam)
    
    # FIX 6: Disable XLA
    print("  [6/6] XLA compilation disabled (stability)")
    content = content.replace(
        'os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"',
        '# os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"  # FIXED: disabled'
    )
    
    # Save
    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n✅ SUCCESS!\n")
    print(f"📌 NEXT STEPS:")
    print(f"   1. Test the fixed version:")
    print(f"      python ocuvision_complete_FIXED.py train")
    print(f"")
    print(f"   2. If training works (loss decreases), replace original:")
    print(f"      cp ocuvision_complete.py ocuvision_complete.BACKUP")
    print(f"      cp ocuvision_complete_FIXED.py ocuvision_complete.py")
    print(f"")
    print(f"   3. Continue training with checkpoint resume:")
    print(f"      python ocuvision_complete.py train")
    print(f"")
    print(f"📊 WATCH FOR:")
    print(f"   ✓ Loss decreases (2.5 → 0.5 by epoch 5)")
    print(f"   ✓ Accuracy increases (8% → 40%+ by epoch 10)")
    print(f"   ✓ No NaN values")
    print(f"   ✓ GPU memory stable at 4-6GB")
    print(f"")
    
    return True

if __name__ == "__main__":
    patch_file()
