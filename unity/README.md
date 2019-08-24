# AnimalAI カステム Unity環境

## 変更点

### TrainingAgent.cs

`Assets/AnimalAIOlympics/TrainEnv/Scripts/TrainingAgent.cs`

```
    public override void CollectObservations()
    {
        Vector3 localVel = transform.InverseTransformDirection(_rigidBody.velocity);
        AddVectorObs(localVel);

        // CHANGED: Custumly added global positaion and rotation
        Vector3 position = _rigidBody.velocity;
        AddVectorObs(position);

        Quaternion rotation = _rigidBody.rotation;
        AddVectorObs(rotation);
    }
```

### BrainのParameter

`TrainEnv/Brains/Learner.asset`と`TrainEnv/Brains/Player.asset`

の`VectorObservations/SpaceSize`を`3`から`10`に変更
