import time
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image


# Defining page settings
st.set_page_config(
    page_title = 'Cassava Leaf Disease',
    page_icon = ':leaves:',
    layout = 'wide',
    initial_sidebar_state = 'expanded',
)

# loadig user image file
uploaded_file = st.sidebar.file_uploader("Drag or upload a JPG file", type = ["jpg"])

# Defining 2 columns for the layout

image = Image.open('C:/Users/Otinwa Ayomide/Documents/Group 5_CASSAVA DISEASE DETECTION/APPLICATION/img/cassavap.jpg')
col1, col2 = st.columns(2)
col1.title('cassava leaf disease')
col1.write('')
col1.header('Predicting system')
col2.image(image)

# Information
with st.expander('Information', expanded = True):
    st.write('''The model aims to classify the type of disease found on cassava leaves. It could be them,
                 Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD)
                 and also identify a healthy leaf. Feel free to test, look for images of the diseases and test.''')
    st.write('The final solution had an average accuracy of 69% in cross-validation.')
st.write('')


def leaf_treatment(img: None) -> None:
    '''
        preparing the images, resizing them to 456*456 and placing them in the shape (1,456,456,3) in a numpy array, serving as input for the model's input layer.
    '''
    leaf = Image.open(uploaded_file).resize((300, 300), resample = Image.NEAREST)
    leaf = np.expand_dims(leaf, axis = 0)
    return leaf


def leaf_predict(leaf: None) -> tuple:
    '''
        performing image prediction based on the array, the result is a softmax function with the
         percentage for each disease, where we will take the highest percentage indicated. thus selecting
         the disease and descriptions that will be uploaded to the page.
    '''
    result = leaf_model.predict(leaf).argmax(axis = 1)
    disease = ''
    description = ''
    causer = ''
    link = ''

    if result == 0:
        disease = 'Cassava Bacterial Blight (CBB) :fallen_leaf:'
        link = 'https://plantix.net/pt/library/plant-diseases/300039/cassava-bacterial-blight'
        description = '''Translated from English - Xanthomonas axonopodis pv. manihotis is the pathogen that causes the bacterial blight of cassava.
                         Originally discovered in Brazil in 1912, the disease followed cassava cultivation around the world.'''
        causer = '''The symptoms are caused by a strain of the bacterium Xanthomonas axonopodis, which readily infects cassava plants.
                     (Manihotis). Within the crop (or fields), bacteria are dispersed by wind or splashing rain. Tools
                     contaminated areas are also an important means of dissemination, as well as the movement of people and animals through the plantations,
                     especially during or after rain. However, the biggest problem with this pathogen is its distribution over long distances.
                     in planting material, cuttings and seeds apparently without symptoms, particularly in Africa and Asia. The process of
                     contamination and disease development requires 12 hours at 90-100% relative humidity, with ideal temperatures ranging
                     from 22 to 30 °C. Bacteria remain viable for many months in stems and gum, resuming activity during periods
                     rainy. The other important host of this bacterium is the ornamental plant Euphorbia pulcherrima (parrot's beak).'''
    elif result == 1:
        disease = 'Cassava Brown Streak Disease (CBSD) :fallen_leaf:'
        link = 'https://plantix.net/pt/library/plant-diseases/200043/cassava-brown-streak-disease'
        description = '''Cassava brown ray virus disease is a harmful disease for cassava plants
                         and it is especially problematic in East Africa. It was first identified in 1936 in Tanzania and spread
                         to other coastal areas of East Africa, from Kenya to Mozambique.'''
        causer = '''The symptoms are caused by cassava brown stripe disease, which as far as is known only affects cassava and maniçoba.
                     (Manihot glaziovii). CBSV can be transmitted by mites and aphids, as well as the whitefly Bemisia tabaci. However, the
                     The predominant way of spreading the disease is by infected cuttings transported by humans and the lack of cleanliness of
                     agricultural tools in the field. Cassava varieties differ greatly in their sensitivity and response to contamination,
                     with yield losses ranging from 18 to 70%, depending on the areas of contamination and environmental conditions. measures of
                     quarantine are necessary to restrict the movement of infected cuttings between countries affected by the disease and those
                     no registry.'''
    elif result == 2:
        disease = 'Cassava Green Mottle (CGM) :fallen_leaf:'
        link = 'https://www.pestnet.org/fact_sheets/cassava_green_mottle_068.htm'
        description = 'Cassava green spot virus is a plant pathogenic virus of the Secoviridae family.'
        causer = '''Severely damaged leaves dry out and fall off, which can cause a characteristic candlestick appearance.
                     Due to reduced plant growth, starch accumulation in storage roots is slowed down, sometimes even reversed,
                     and root yield losses in the absence of any control measures can reach 50%. where the leaves are eaten
                     as vegetables by farmers, a corresponding loss occurs. Reduced growth and stunting of the tips also
                     are responsible for twisted and thin stems, damaging the planting material to be used in the next season.
                     The size of CGM populations, and therefore yield losses, are generally influenced by several factors,
                     including: (1) host plant age - young plants are more exposed and susceptible to CGM attacks
                     than older plants; (2) season - the severity of damage is greater during the dry season than the wet season, and rainfall
                     strong temperatures can reduce CGM populations, (3) temperature - populations increase with increasing temperature, leading to
                     sometimes to a very rapid increase in populations and damage, and (4) poor agronomic practices - plants grown in soils
                     poor are more susceptible to attack by mites.'''
    elif result == 3 :
        disease = 'Cassava Mosaic Disease (CMD) :fallen_leaf:'
        link = 'https://plantix.net/pt/library/plant-diseases/200042/cassava-mosaic-disease'
        description = '''Translated from English - Cassava mosaic virus is the common name used to refer to any of the eleven
                         different species of phytopathogenic viruses of the genus Begomovirus.'''
        causer = '''The symptoms of African cassava mosaic are caused by a group of viruses that frequently infect
                     parallel to cassava plants. These viruses can be persistently transmitted by the whitefly Bemisia tabaci,
                     as well as by cuttings derived from infected planting material. Whiteflies are carried by currents
                     of wind and can spread the virus over distances of several kilometers. Cassava varieties differ greatly in their
                     susceptibility to the virus, but usually young leaves are the first to show symptoms, as whiteflies
                     prefer to feed on new tissue. The distribution of the virus is very dependent on the population of this insect, which in turn
                     time is subject to prevailing weather conditions. If large populations of whiteflies coincide with conditions
                     ideal for cassava development, the virus will spread quickly. The temperatures preferred by this pest are
                     estimated between 20 to 32 °C.'''
    elif result == 4:
        disease = 'healthy :leaves:'
        description = 'We did not identify the presence of disease in the image. :thumbsup:'
    else:
        print('Unable to perform identification')
    
    return result, disease, description, causer, link


if __name__ == "__main__":
    # Page to be displayed with the results
    if uploaded_file is not None:

        # Loading CNN model
        #leaf_model = load_model('CassavaLeafDisease.h5')
        leaf_model = load_model('cassava_leaf.h5')

        leaf = leaf_treatment(uploaded_file)
        leaf = leaf_predict(leaf)
        img = Image.open(uploaded_file)
        col3, col4 = st.columns(2)
        col3.write('')
        col3.image(img)
        col4.subheader(leaf[1])
        col4.write(leaf[2])
        col4.write(leaf[3])

        if leaf[0] != 4:
            col4.markdown('**See:**')
            col4.write(leaf[4])
