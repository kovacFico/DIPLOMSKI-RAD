import numpy as np


class ColorModel:
    '''
    Online appearance model za dominantnu boju rukavice. 

    Model koristi:
        - reduced feature vector
        - fixed covariance
        - Mahalanobis gating
        - EMA adaptation

    Feature vector:
        [
            mean_A,
            mean_B,
            std_A,
            std_B
        ]
        
    Args:
        ema_alpha (float): EMA faktor za ažuriranje modela.
        gating_threshold (float): Prag za Mahalanobis gating.
    
    State:
        mean (np.ndarray): Trenutni mean feature vektor.
        cov (np.ndarray): Fiksna kovariancijska matrica.
        inv_cov (np.ndarray): Inverz kovariancijske matrice.
        model_updates (int): Broj validnih updateova.
        is_initialized (bool): Zastavica inicijalizacije modela.
        
    Methods:
        distance(feature): Računa Mahalanobis udaljenost između featurea i modela
        is_valid(feature): Provjerava je li feature unutar gating praga
        update(feature): Ažurira model s novim featureom koristeći EMA
        confidence(): Vraća confidence modela na temelju broja validnih updateova
    '''

    def __init__(self, ema_alpha=0.1, gating_threshold=4.0):
        self.mean = None   
        self.cov = np.diag([
            25.0,  
            25.0,  
            9.0,   
            9.0    
        ])
        self.inv_cov = np.linalg.inv(self.cov)

        self.ema_alpha = ema_alpha
        self.gating_threshold = gating_threshold
        
        self.model_updates = 0
        self.is_initialized = False


    def distance(self, feature):
        '''
        Računa Mahalanobis udaljenost između featurea i trenutnog modela. Mahalanobis
        udaljenost se mjeri kao kvadratni korijen od transponiranog razlike između featurea 
        i mean, pomnožene s inverzom kovariancijske matrice i samom razlikom, tj u formuli: 
            sqrt((feature - mean)^T * inv_cov * (feature - mean)). 
        Ako model nije inicijaliziran ili je feature None, vraća None.
        
        Args:
            feature (np.ndarray): Feature vektor za koji se računa udaljenost.
            
        Returns:
            float or None: Mahalanobis udaljenost između featurea i modela, 
                    ili None ako model nije inicijaliziran ili je feature None.
        '''

        if (
            not self.is_initialized or
            feature is None
        ):
            return None

        diff = feature - self.mean

        dist = np.sqrt(
            diff.T @ self.inv_cov @ diff
        )

        return float(dist)


    def is_valid(self, feature):
        '''
        Provjerava je li feature dovoljno blizu trenutnom modelu kako bi se mogao 
        koristiti za update. Koristi Mahalanobis udaljenost i gating threshold.
        '''

        if not self.is_initialized:
            return True

        dist = self.distance(feature)

        return dist < self.gating_threshold


    def update(self, feature):
        '''
        Online EMA update appearance modela na temelju novih featurea. Ako je 
        feature dovoljno blizu trenutnog modela, model se ažurira. Ako model 
        nije inicijaliziran, inicijalizira se s prvim featureom. Povratna 
        vrijednost označava je li update bio validan i je li model ažuriran.
        
        
        Args:
            feature (np.ndarray): Novi feature vektor koji se koristi za update.
            
        Returns:
            bool: True ako je update bio validan i model je ažuriran, False inače.
        '''

        if feature is None:
            return False

        feature = np.asarray(feature, dtype=np.float32)

        if self.mean is None:
            self.mean = feature.copy()
            self.model_updates = 1
            self.is_initialized = True
            return True

        if not self.is_valid(feature):
            return False



        alpha = self.ema_alpha
        self.mean = (
            (1 - alpha) * self.mean +
            alpha * feature
        )
        self.model_updates += 1

        return True


    def confidence(self):
        '''
        Confidence vrijednost koja raste s brojem validnih updateova.
        '''

        if not self.is_initialized:
            return 0.0

        return min(self.model_updates / 20.0, 1.0)
    
    
    def model_distance(self, other):
        '''
        Računa Mahalanobis udaljenost između dva modela. Udaljenost se koristi
        za dobivanje vrijednosti cost matrice Trackera u multiview praćenju.    
        
        Args:
            other (ColorModel): Drugi ColorModel s kojim se računa udaljenost.
            
        Returns:
            float: Mahalanobis udaljenost između dva modela.
        '''
        delta = self.mean - other.mean
        cov = self.cov + other.cov

        return np.sqrt(
            delta.T @ np.linalg.inv(cov) @ delta
        )