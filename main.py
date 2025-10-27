import synth_ai


def main() -> None:
    helper = getattr(synth_ai, "help", None)
    if callable(helper):
        print(helper())
    else:
        print("Synth AI CLI")


if __name__ == "__main__":
    main()
