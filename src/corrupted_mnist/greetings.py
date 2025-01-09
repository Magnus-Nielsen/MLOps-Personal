import typer

app = typer.Typer()


@app.command()
def hello(count: int = 1):
    for _ in range(count):
        print("Hello world!")


if __name__ == "__main__":
    app()
