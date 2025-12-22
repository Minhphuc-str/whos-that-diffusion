import argparse
import subprocess
import os

def run_command(command):
    try:
        # We use shell=True to handle the environment variables correctly
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running step: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to silhouette (e.g., test_dog.png)")
    parser.add_argument("--output", type=str, default="final_pokemon.png", help="Output filename")
    args = parser.parse_args()

    print("="*40)
    print("🚀 STARTING POKEMON GENERATION PIPELINE")
    print("="*40)

    # Step 1: Generate the Image
    print("\n🎨 Phase 1: Dreaming up the creature...")
    # Note: We enforce the CPU fallback here for safety
    gen_cmd = (
        f"export PYTORCH_ENABLE_MPS_FALLBACK=1 && "
        f"python inference_cnet.py --input '{args.input}' --output '{args.output}'"
    )
    run_command(gen_cmd)

    # Step 2: Name the Creature
    print("\n🧠 Phase 2: Analyzing and Naming...")
    name_cmd = f"python auto_namer.py --image '{args.output}'"
    run_command(name_cmd)

    print("\n" + "="*40)
    print(f"✅ DONE! Meet your new Pokemon at: {args.output}")
    print("="*40)

if __name__ == "__main__":
    main()