"""Resume the official MHNet archive download, validating HTTP byte ranges."""
import argparse
from pathlib import Path
import urllib.request
import time
import signal

URL = 'https://files-zillowstatic-com.s3.us-west-2.amazonaws.com/research/public/StaticFiles/salve/data/ZInD_HorizonNet_predictions.tar.gz'
SIZE = 732377373


def timeout_handler(signum, frame):
    raise TimeoutError('Range request exceeded 60 seconds')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    p.add_argument('--direct', action='store_true', help='Bypass proxy for the public S3 download')
    args = p.parse_args()
    signal.signal(signal.SIGALRM, timeout_handler)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.direct else urllib.request.build_opener()
    target = args.destination
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != SIZE:
            raise RuntimeError('Existing archive size mismatch')
        return
    part = target.with_name(target.name + '.part')
    failures = 0
    while failures < 5:
        offset = part.stat().st_size if part.exists() else 0
        if offset == SIZE:
            part.rename(target)
            print('Complete:', target, flush=True)
            return
        if offset > SIZE:
            raise RuntimeError('Partial file exceeds expected size')
        end = min(SIZE - 1, offset + 1024 * 1024 - 1)
        try:
            signal.alarm(60)
            request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{end}'})
            with opener.open(request, timeout=30) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{end}/{SIZE}':
                    raise RuntimeError('Server did not honor requested byte range')
                data = response.read()
                if len(data) != end - offset + 1:
                    raise RuntimeError('Incomplete range response')
            signal.alarm(0)
            with part.open('ab') as stream:
                stream.write(data)
            print(f'{end + 1}/{SIZE}', flush=True)
            failures = 0
        except Exception as exc:
            signal.alarm(0)
            failures += 1
            print(f'Retry: {exc}', flush=True)
            time.sleep(2)
    raise RuntimeError('Download incomplete; rerun to resume')


if __name__ == '__main__':
    main()
