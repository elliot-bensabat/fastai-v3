import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1E-klKePmFri4sapmMwM36WhxLgUXf6dN'
export_file_name = 'dogclassifier.pkl'

classes = ['Afghan_hound',
 'African_hunting_dog',
 'Airedale',
 'American_Staffordshire_terrier',
 'Appenzeller',
 'Australian_terrier',
 'Bedlington_terrier',
 'Bernese_mountain_dog',
 'Blenheim_spaniel',
 'Border_collie',
 'Border_terrier',
 'Boston_bull',
 'Bouvier_des_Flandres',
 'Brabancon_griffon',
 'Brittany_spaniel',
 'Cardigan',
 'Chesapeake_Bay_retriever',
 'Chihuahua',
 'Dandie_Dinmont',
 'Doberman',
 'English_foxhound',
 'English_setter',
 'English_springer',
 'EntleBucher',
 'Eskimo_dog',
 'French_bulldog',
 'German_shepherd',
 'Gordon_setter',
 'Great_Dane',
 'Great_Pyrenees',
 'Greater_Swiss_Mountain_dog',
 'Ibizan_hound',
 'Irish_setter',
 'Irish_terrier',
 'Irish_water_spaniel',
 'Irish_wolfhound',
 'Italian_greyhound',
 'Japanese_spaniel',
 'Kerry_blue_terrier',
 'Labrador_retriever',
 'Lakeland_terrier',
 'Leonberg',
 'Lhasa',
 'Maltese_dog',
 'Mexican_hairless',
 'Newfoundland',
 'Norfolk_terrier',
 'Norwegian_elkhound',
 'Norwich_terrier',
 'Old_English_sheepdog',
 'Pekinese',
 'Pembroke',
 'Pomeranian',
 'Rhodesian_ridgeback',
 'Rottweiler',
 'Saint_Bernard',
 'Saluki',
 'Samoyed',
 'Scotch_terrier',
 'Scottish_deerhound',
 'Sealyham_terrier',
 'Shetland_sheepdog',
 'Siberian_husky',
 'Staffordshire_bullterrier',
 'Sussex_spaniel',
 'Tibetan_mastiff',
 'Tibetan_terrier',
 'Tzu',
 'Walker_hound',
 'Weimaraner',
 'Welsh_springer_spaniel',
 'West_Highland_white_terrier',
 'Yorkshire_terrier',
 'affenpinscher',
 'basenji',
 'basset',
 'beagle',
 'bloodhound',
 'bluetick',
 'borzoi',
 'boxer',
 'briard',
 'bull_mastiff',
 'cairn',
 'chow',
 'clumber',
 'coated_retriever',
 'coated_wheaten_terrier',
 'cocker_spaniel',
 'collie',
 'dhole',
 'dingo',
 'giant_schnauzer',
 'golden_retriever',
 'groenendael',
 'haired_fox_terrier',
 'haired_pointer',
 'keeshond',
 'kelpie',
 'komondor',
 'kuvasz',
 'malamute',
 'malinois',
 'miniature_pinscher',
 'miniature_poodle',
 'miniature_schnauzer',
 'n02099429-curly-coated_retriever',
 'otterhound',
 'papillon',
 'pug',
 'redbone',
 'schipperke',
 'silky_terrier',
 'standard_poodle',
 'standard_schnauzer',
 'tan_coonhound',
 'toy_poodle',
 'toy_terrier',
 'vizsla',
 'whippet']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
