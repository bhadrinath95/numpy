# Generated by Django 3.0.8 on 2020-07-04 06:30

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Topic',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('heading', models.CharField(max_length=120)),
                ('created_date', models.DateField()),
                ('urlpath', models.URLField()),
            ],
        ),
    ]
