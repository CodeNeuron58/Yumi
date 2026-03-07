from Yumi_Hears.pipeline import AudioPipeline

def main():
    pipeline = AudioPipeline()
    while True:
        try:
            pipeline.run_cycle()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
