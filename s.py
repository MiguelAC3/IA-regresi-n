import matplotlib.pyplot as plt

def dibujar_red_neuronal(capas):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Configuración visual
    left, right, bottom, top = .1, .9, .1, .9
    layer_sizes = capas
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Colores por capa (Entrada, Oculta 1, Oculta 2, Salida)
    colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71'] # Azul, Rojo, Amarillo, Verde
    labels = ['Capa de Entrada\n(7 datos)', 'Capa Oculta 1\n(8 neuronas)', 'Capa Oculta 2\n(4 neuronas)', 'Salida\n(Predicción)']

    # Recorrer capas y nodos
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        
        # Dibujar nodos (círculos)
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color=colors[n], ec='k', zorder=4)
            ax.add_artist(circle)
            
            # Anotación de neurona (opcional)
            # ax.text(n*h_spacing + left, layer_top - m*v_spacing, str(m+1), ha='center', va='center', zorder=5)

        # Dibujar líneas (conexiones) con la capa siguiente
        if n < len(layer_sizes) - 1:
            next_layer_size = layer_sizes[n+1]
            next_layer_top = v_spacing*(next_layer_size - 1)/2. + (top + bottom)/2.
            
            for m in range(layer_size):
                for o in range(next_layer_size):
                    line = plt.Line2D([n*h_spacing + left, (n+1)*h_spacing + left],
                                      [layer_top - m*v_spacing, next_layer_top - o*v_spacing], c='gray', alpha=0.3)
                    ax.add_artist(line)
        
        # Poner nombre de la capa arriba
        ax.text(n*h_spacing + left, 0.95, labels[n], ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title("Arquitectura de tu Modelo de Personalidad", fontsize=15)
    
    # Guardar y mostrar
    plt.savefig('arquitectura_red.png', dpi=300, bbox_inches='tight')
    print("Imagen 'arquitectura_red.png' generada exitosamente.")
    plt.show()

# Tu arquitectura exacta: 
# 7 entradas -> 8 neuronas -> 4 neuronas -> 1 salida
capas_modelo = [7, 8, 4, 1]

dibujar_red_neuronal(capas_modelo)