import asyncio
import os
import aiohttp
from lxml.html import fromstring


async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()


async def fetch_file(session, url):

    async with session.get(url) as response:
        return await response.read()


def write_to_file(filename, content, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    f = open(folder_name + filename, 'wb')
    f.write(content)
    f.close()


async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'http://www.cs.put.poznan.pl/elukasik/PIRD18/Projekty/Projekt_1/komendy/')
        html = html[html.find("<html>"):].replace("<hr>", "")
        print(html)
        tree = fromstring(html)
        links = tree.xpath(".//pre/a/text()")
        results = await asyncio.gather(
            *[fetch(session, 'http://www.cs.put.poznan.pl/elukasik/PIRD18/Projekty/Projekt_1/komendy/' + link) for link in links[-6:]],
            return_exceptions=True)
        for result, link in zip(results, links[-6:]):
            tree = fromstring(result)
            for a in tree.xpath(".//img/following-sibling::a/text()"):
                if ".WAV" in a:
                    lul = await fetch_file(session, 'http://www.cs.put.poznan.pl/elukasik/PIRD18/Projekty/Projekt_1/komendy/' + link + a)
                    print(a)
                    write_to_file(a, lul, link)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
