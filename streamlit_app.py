""" Streamlit app """
import streamlit as st
from pathlib import Path
import base64

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
    Y = 2*np.fft.fft(y)/n  # fft computing and normalization
    Y = Y[range(int(n/2))]
    return (frq, Y)





def main():
    # cs_sidebar()
    cs_body()

    return None

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

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