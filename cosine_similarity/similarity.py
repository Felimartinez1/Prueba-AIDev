import math

def cosine_similarity(vector1, vector2):
    """
    Calcula la similitud del coseno entre dos vectores.

    Args:
        vector1 (list): Primer vector de valores numericos.
        vector2 (list): Segundo vector de valores numericos.

    Returns:
        float: Similitud del coseno entre los dos vectores, un valor entre -1 y 1.
    """

    if len(vector1) != len(vector2):
        raise ValueError("Los vectores deben tener la misma longitud")

    dot_product = sum([a * b for a, b in zip(vector1, vector2)])
    norm1 = math.sqrt(sum([a ** 2 for a in vector1]))
    norm2 = math.sqrt(sum([b ** 2 for b in vector2]))

    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity

vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

similarity = cosine_similarity(vector1, vector2)
print("{:.4f}".format(similarity))
