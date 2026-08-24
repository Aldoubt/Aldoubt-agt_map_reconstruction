from pathlib import Path


def write_report(output_dir: str, images: dict):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    html = [
        '<html><head><title>AGT Map Reconstruction Benchmark</title></head><body>',
        '<h1>Agricultural LiDAR Map Reconstruction Benchmark</h1>'
    ]

    for name, image in images.items():
        html.append(f'<h2>{name}</h2>')
        html.append(f'<img src="{image}" width="800">')

    html.append('</body></html>')
    (out / 'benchmark_report.html').write_text('\n'.join(html))
