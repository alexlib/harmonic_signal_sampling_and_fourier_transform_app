"""
Streamlit Cheat Sheet

App to summarise streamlit docs v1.8.0

There is also an accompanying png and pdf version

https://github.com/daniellewisDL/streamlit-cheat-sheet

v1.8.0 October 2021

Author:
    @daniellewisDL : https://github.com/daniellewisDL

Contributors:
    @arnaudmiribel : https://github.com/arnaudmiribel
    @akrolsmir : https://github.com/akrolsmir
    @nathancarter : https://github.com/nathancarter

"""

import streamlit as st
from pathlib import Path
import base64

from scipy import fft
import numpy as np
import matplotlib.pyplot as plt


# Initial page config

st.set_page_config(
     page_title='Streamlit cheat sheet',
     layout="wide",
     initial_sidebar_state="expanded",
)

def spectrum(y, Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of a sampled
    signal y(t), sampling frequency Fs (length of a signal
    provides the number of samples recorded)

    Following: http://goo.gl/wRoUn
    """
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T  # two sides frequency range
    frq = frq[range(np.int(n/2))]  # one side frequency range
    Y = 2*fft.fft(y)/n  # fft computing and normalization
    Y = Y[range(np.int(n/2))]
    return (frq, Y)





def main():
    cs_sidebar()
    cs_body()

    return None

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# sidebar

def cs_sidebar():


    # N = st.sidebar.checkbox('Streamlines?',value = False)

    # colorbar = st.sidebar.checkbox('Color bar?',value=False)

    # average = st.sidebar.checkbox('Average?',value=False)



#     st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=32 height=32>](https://streamlit.io/)'''.format(img_to_bytes("logomark_website.png")), unsafe_allow_html=True)
#     st.sidebar.header('Streamlit cheat sheet')

#     st.sidebar.markdown('''
# <small>Summary of the [docs](https://docs.streamlit.io/en/stable/api.html), as of [Streamlit v1.8.0](https://www.streamlit.io/).</small>
#     ''', unsafe_allow_html=True)

#     st.sidebar.markdown('__How to install and import__')

#     st.sidebar.code('$ pip install streamlit')

#     st.sidebar.markdown('Import convention')
#     st.sidebar.code('>>> import streamlit as st')

#     st.sidebar.markdown('__Add widgets to sidebar__')
#     st.sidebar.code('''
# st.sidebar.<widget>
# >>> a = st.sidebar.radio(\'R:\',[1,2])
#     ''')

#     st.sidebar.markdown('__Command line__')
#     st.sidebar.code('''
# $ streamlit --help
# $ streamlit run your_script.py
# $ streamlit hello
# $ streamlit config show
# $ streamlit cache clear
# $ streamlit docs
# $ streamlit --version
#     ''')

#     st.sidebar.markdown('__Pre-release features__')
#     st.sidebar.markdown('[Beta and experimental features](https://docs.streamlit.io/library/advanced-features/prerelease#beta-and-experimental-features)')
#     st.sidebar.code('''
# pip uninstall streamlit
# pip install streamlit-nightly --upgrade
#     ''')

#     st.sidebar.markdown('''<small>[st.cheat_sheet v1.8.0](https://github.com/daniellewisDL/streamlit-cheat-sheet)  | April 2022</small>''', unsafe_allow_html=True)

    return None

##########################
# Main body of cheat sheet
##########################

def cs_body():
    st.title("Harmonic signal sampling and Fourier transform app")

    st.subheader('Sample signal and its spectrum')

    N = st.sidebar.slider('Number of samples, N', min_value=64, max_value=1024, step=64)
    ff = st.sidebar.slider('Signal frequency (Hz) ',  min_value=1, max_value=15)
    fs = st.sidebar.slider('Sampling frequency (Hz)',min_value=5, max_value=150)
    A = st.sidebar.slider('Amplitude [V]',min_value=1, max_value=10)

    T = N/fs  # sampling period
    t = np.arange(0.0, T, T/N)  # sampling time steps
    y = A*np.sin(2*np.pi*ff*t)  # sampled signal
    frq, Y = spectrum(y, fs)  # FFT(sampled signal)

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(t, y, 'b.')
    ax[0].set_xlabel('$t$ [s]')
    ax[0].set_ylabel('Y [V]')
    ax[0].plot(t[0], y[0], 'ro')
    ax[0].plot(t[-1], y[-1], 'ro')

    ax[1].plot(frq, abs(Y), 'r')  # plotting the spectrum
    ax[1].set_xlabel('$f$ (Hz)')
    ax[1].set_ylabel('$|Y(f)|$')
    lgnd = str(f'N = {N}, f = {fs} Hz')
    ax[1].legend([lgnd], loc='best')

    st.pyplot(fig)


    return None

# Run main()

if __name__ == '__main__':
    main()