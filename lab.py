import argparse
import ollama

def main():
    parser = argparse.ArgumentParser(description="Genera text amb el model gemma3:4b.")
    parser.add_argument("prompt", help="El text del prompt (obligatori)")
    parser.add_argument("--vegades", type=int, default=5, help="El nombre de vegades a generar el text (per defecte: 5)")
    parser.add_argument("--temperature", type=float, default=0.8, help="La temperatura per a la generació (per defecte: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="El seed per a la generació (per defecte: 42)")

    args = parser.parse_args()

    # Imprimir opcions
    print(f"Prompt: {args.prompt}")
    print(f"vegades: {args.vegades}")
    print(f"temperature: {args.temperature}")
    print(f"seed: {args.seed}")


    for i in range(args.vegades):
        try:
            response = ollama.generate(
                model="gemma3:4b",
                prompt=args.prompt,
                options={
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "top_k": 50,
                    "top_p": 0.95,
                    "num_predict": 256  # Ajustat per a la majoria de les respostes
                }
            )
            print(f"{i+1}. {response['response']}")

        except Exception as e:
            print(f"Error durant la generació a la iteració {i+1}: {e}")

if __name__ == "__main__":
    main()