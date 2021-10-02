from flask import Flask, render_template, request, redirect, session, flash, send_from_directory, url_for
from flask.helpers import url_for
from PIL import Image
import tensorflow as tf
import keras
import os, time
import numpy as np

app = Flask(__name__)

app.config['MODEL_PATH'] = os.path.dirname(os.path.abspath(__file__)) + '/models'
app.config['PHOTO_PATH'] = os.path.dirname(os.path.abspath(__file__)) + '/photo'
nome_foto = 'foto_atual'

model_file_inception = app.config['MODEL_PATH'] + '/trained_model_inception.h5'
model_file_xception = app.config['MODEL_PATH'] + '/trained_model_xception.h5'

#saved_model = keras.models.load_model(model_file_inception)
saved_model = keras.models.load_model(model_file_xception)

classes = [
    'Chihuahua' ,
    'Spaniel japonês' ,
    'Maltês' ,
    'Pequinês' ,
    'Shih-Tzu' ,
    'Cavalier king charles spaniel' ,
    'Papillon' ,
    'Toy terrier' ,
    'Rhodesian ridgeback' ,
    'Galgo afegão' ,
    'Basset' ,
    'Beagle' ,
    'Bloodhound' ,
    'Bluetick' ,
    'Coonhound preto e castanho' ,
    'Walker hound' ,
    'Foxhound-inglês' ,
    'Redbone' ,
    'Borzoi' ,
    'Irish wolfhound' ,
    'Galguinho italiano' ,
    'Whippet' ,
    'Ibizan hound' ,
    'Elkhound' ,
    'Otterhound' ,
    'Saluki' ,
    'Deerhound' ,
    'Weimaraner' ,
    'Staffordshire bull terrier' ,
    'American Staffordshire terrier' ,
    'Bedlington terrier' ,
    'Border terrier' ,
    'Kerry blue terrier' ,
    'Terrier irlandês' ,
    'Norfolk terrier' ,
    'Norwich terrier' ,
    'Yorkshire terrier' ,
    'Fox terrier de pelo duro' ,
    'Lakeland terrier' ,
    'Sealyham terrier' ,
    'Airedale terrier' ,
    'Cairn terrier' ,
    'Terrier Australiano' ,
    'Dandie dinmont terrier' ,
    'Boston terrier' ,
    'Schnauzer miniatura' ,
    'Schnauzer gigante' ,
    'Schnauzer standard' ,
    'Scotch terrier' ,
    'Terrier tibetano' ,
    'silky terrier' ,
    'Soft coated wheaten terrier' ,
    'West highland white terrier' ,
    'Lhasa apso' ,
    'Flat-coated retriever' ,
    'Curly coated retriever' ,
    'Golden retriever' ,
    'Labrador retriever' ,
    'Chesapeake bay retriever' ,
    'Braco alemão de pelo curto' ,
    'Vizsla' ,
    'Setter inglês' ,
    'Setter irlandês' ,
    'Setter gordon' ,
    'Spaniel bretão' ,
    'Clumber spaniel' ,
    'Springer spaniel inglês' ,
    'Springer spaniel de Gales' ,
    'Cocker spaniel' ,
    'Sussex spaniel' ,
    'Cão dágua irlandês' ,
    'Pastor Húngaro' ,
    'Schipperke' ,
    'Pastor Belga Groenendael' ,
    'Pastor-belga-malinois' ,
    'Pastor-de-brie' ,
    'Kelpie australiano' ,
    'Komondor' ,
    'Old english sheepdog' ,
    'Pastor-de-shetland' ,
    'Collie' ,
    'Border collie' ,
    'Boiadeiro da Flandres' ,
    'Rottweiler' ,
    'Pastor-alemão' ,
    'Doberman' ,
    'Pinscher miniatura' ,
    'Grande boiadeiro suíço' ,
    'Bernese' ,
    'Boiadeiro de Appenzell' ,
    'Boiadeiro de Entlebuch' ,
    'Boxer' ,
    'Bulmastife' ,
    'Mastim tibetano' ,
    'Buldogue francês' ,
    'Dogue alemão' ,
    'São-bernardo' ,
    'Cão Esquimó Canadense' ,
    'Malamute do Alasca' ,
    'Husky siberiano' ,
    'Affenpinscher' ,
    'Basenji' ,
    'Pug' ,
    'Leonberger' ,
    'Terra-nova' ,
    'Cão de montanha dos Pirenéus' ,
    'Samoieda' ,
    'Spitz-alemão-anão' ,
    'Chow-chow' ,
    'Keeshond' ,
    'Petit Brabançon' ,
    'Welsh corgi pembroke' ,
    'Welsh corgi cardigan' ,
    'Poodle toy' ,
    'Poodle miniatura' ,
    'Poodle gigante' ,
    'Pelado-mexicano' ,
    'Dingo' ,
    'Cão-selvagem-asiático' ,
    'Mabeco' ,
]

def recupera_imagem():
    for nome_arquivo in os.listdir(app.config['PHOTO_PATH']):
        if nome_foto in nome_arquivo:
            return nome_arquivo


def deleta_arquivo():
    arquivo = recupera_imagem()
    os.remove(os.path.join(app.config['PHOTO_PATH'], arquivo))


@app.route('/', methods=['POST', 'GET',])
def index():
    return render_template('inicial_file.html', titulo='Doguíneos API')

@app.route('/resultado')
def novo():
    return render_template('resultado.html', titulo='ops')


@app.route('/analisar', methods=['POST',])
def analisar():
    arquivo_img = request.files['arquivo']
    if (arquivo_img):
        upload_path = app.config['PHOTO_PATH']
        timestamp = time.time()

        nome_arquivo = nome_foto + '-' + str(timestamp) + '.jpg'
        deleta_arquivo()
        arquivo_img.save(upload_path + '/' + nome_arquivo)
    
        img = Image.open(arquivo_img.stream).convert("RGB")
        img = img.resize((299,299), Image.ANTIALIAS)
        img_dog = np.asarray(img)/255
        img_dog = img_dog.reshape(1,299,299,3)
        predicted_dog = saved_model.predict(img_dog)
        predictedindex = predicted_dog.argsort(axis=-1)

        classe = classes[predictedindex[0][-1]]
        porcentagem = float(predicted_dog[0][predictedindex[0][-1]]*100)
        resultado_texto = ''

        if (porcentagem > 50):
            resultado_texto = 'Esse cachorro provavelmente é da raça ' + classe + '.'
        else:
            resultado_texto = 'Não conseguimos identificar com precisão a raça desse cachorro. Veja a melhor previsão abaixo:'

        outros = ['' + classes[predictedindex[0][-2]] + ': ' + '{0:.2f}'.format(float(predicted_dog[0][predictedindex[0][-2]]*100))
                            , '' + classes[predictedindex[0][-3]] + ': ' + '{0:.2f}'.format(float(predicted_dog[0][predictedindex[0][-3]]*100))
                            , '' + classes[predictedindex[0][-4]] + ': ' + '{0:.2f}'.format(float(predicted_dog[0][predictedindex[0][-4]]*100))]
        
        return render_template('resultado.html', titulo='Resultado', texto=resultado_texto, classe=classe
                           , foto_atual=nome_arquivo, porcentagem='{0:.2f}'.format(porcentagem), outros=outros )
    
    return redirect(url_for('index'))


@app.route('/photo/<nome_arquivo>')
def imagem(nome_arquivo):
    return send_from_directory('photo', nome_arquivo)


app.run()