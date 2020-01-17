# AnimalAI Custom Unity Environment

## Unity version

We used unity version `2018.3.13f1`.



## Where we changed

The default `vector_observations` in the `BrainInfo` has only local velocity `(vx, vy, vz)`, but we added agent's absolute position and angle information.


### TrainingAgent.cs

`Assets/AnimalAIOlympics/TrainEnv/Scripts/TrainingAgent.cs`

Added global postion (x,y,z), angle (0~360 degree), and LIDAR distance and target id info.


```
    public override void CollectObservations()
    {
        Vector3 localVel = transform.InverseTransformDirection(_rigidBody.velocity);
        AddVectorObs(localVel);

        // CHANGED: Custumly added global positaion, rotation and LIDAR info
        Vector3 position = _rigidBody.position;
        AddVectorObs(position);

        float rotation = transform.eulerAngles.y;
        AddVectorObs(rotation);
        
        // Add Lidar scan result
        _lidar.Process(transform);
        AddVectorObs(_lidar.Result);        
    }
```

### LIDAR processing

Please see code around [here](https://github.com/miyosuda/animalai/blob/c191e2a91aef34acffd2dc3ea6612ad04017ee3f/unity/CustomAnimalAI-Environment/Assets/AnimalAIOlympics/TrainEnv/Scripts/TrainingAgent.cs#L179-L302
)


### Brain Parameter

We need to change the parameter of the brain object after adding vector observation info.

Open `Assets/AnimalAIOlympics/TrainEnv/Brains/Learner.asset` and `Assets/AnimalAIOlympics/TrainEnv/Brains/Player.asset` 's property setting and change `VectorObservations/SpaceSize` parameter from `3` to `17`.

17 = 3(local velocity) + 3(position) + 1(rotation) + 10(lidar distance and taraget id)





### Fast boot (Requires Unity Pro license)

The environment app boot time can be reduced by removing splash logo screen. You can do it by turning off the option at

`BuildSetting` -> `PlayerSettings` -> `SplashImage` -> `Splash Screen` -> `Show Splash Screen` 

